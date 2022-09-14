#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>

#include "data_spec.hpp"
#include "common.cuh"
#include "data_spec_packed.cuh"

namespace {

// Automatically choose number of CUDA threads based on HW CUDA kernel count
int cuda_n_threads = -1;
__host__ void auto_cuda_threads() {
    if (~cuda_n_threads) return;
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    const int n_cores = get_sp_cores(dev_prop);
    // Optimize number of CUDA threads per block
    if (n_cores < 2048) {
        cuda_n_threads = 256;
    } if (n_cores < 8192) {
        cuda_n_threads = 512;
    } else {
        cuda_n_threads = 1024;
    }
}

namespace device {
// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __constant__ const float C4[] = {
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
};

#define _SOFTPLUS_M1(x) (logf(1 + expf((x) - 1)))
#define _SIGMOID(x) (1 / (1 + expf(-(x))))

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm(
                scalar_t* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _normalize(
                scalar_t* dir) {
    scalar_t norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _dot3(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

// Calculate basis functions depending on format, for given view directions
template <typename scalar_t>
__device__ __inline__ void maybe_precalc_basis(
    const int format,
    const int basis_dim,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        extra,
    const scalar_t* __restrict__ dir,
    scalar_t* __restrict__ out) {
    switch(format) {
        case FORMAT_ASG:
            {
                // UNTESTED ASG
                for (int i = 0; i < basis_dim; ++i) {
                    const auto& ptr = extra[i];
                    scalar_t S = _dot3(dir, &ptr[8]);
                    scalar_t dot_x = _dot3(dir, &ptr[2]);
                    scalar_t dot_y = _dot3(dir, &ptr[5]);
                    out[i] = S * expf(-ptr[0] * dot_x * dot_x
                                      -ptr[1] * dot_y * dot_y) / basis_dim;
                }
            }  // ASG
            break;
        case FORMAT_SG:
            {
                for (int i = 0; i < basis_dim; ++i) {
                    const auto& ptr = extra[i];
                    out[i] = expf(ptr[0] * (_dot3(dir, &ptr[1]) - 1.f)) / basis_dim;
                }
            }  // SG
            break;
        case FORMAT_SH:
            {
                out[0] = C0;
                const scalar_t x = dir[0], y = dir[1], z = dir[2];
                const scalar_t xx = x * x, yy = y * y, zz = z * z;
                const scalar_t xy = x * y, yz = y * z, xz = x * z;
                switch (basis_dim) {
                    case 25:
                        out[16] = C4[0] * xy * (xx - yy);
                        out[17] = C4[1] * yz * (3 * xx - yy);
                        out[18] = C4[2] * xy * (7 * zz - 1.f);
                        out[19] = C4[3] * yz * (7 * zz - 3.f);
                        out[20] = C4[4] * (zz * (35 * zz - 30) + 3);
                        out[21] = C4[5] * xz * (7 * zz - 3);
                        out[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
                        out[23] = C4[7] * xz * (xx - 3 * yy);
                        out[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
                        [[fallthrough]];
                    case 16:
                        out[9] = C3[0] * y * (3 * xx - yy);
                        out[10] = C3[1] * xy * z;
                        out[11] = C3[2] * y * (4 * zz - xx - yy);
                        out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                        out[13] = C3[4] * x * (4 * zz - xx - yy);
                        out[14] = C3[5] * z * (xx - yy);
                        out[15] = C3[6] * x * (xx - 3 * yy);
                        [[fallthrough]];
                    case 9:
                        out[4] = C2[0] * xy;
                        out[5] = C2[1] * yz;
                        out[6] = C2[2] * (2.0 * zz - xx - yy);
                        out[7] = C2[3] * xz;
                        out[8] = C2[4] * (xx - yy);
                        [[fallthrough]];
                    case 4:
                        out[1] = -C1 * y;
                        out[2] = C1 * z;
                        out[3] = -C1 * x;
                }
            }  // SH
            break;

        default:
            // Do nothing
            break;
    }  // switch
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template <typename scalar_t>
__device__ __inline__ void _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax) {
    // Intersect unit AABB
    scalar_t t1, t2;
    *tmin = 0.0f;
    *tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * invdir[i];
        t2 = t1 +  invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

template <typename scalar_t>
__device__ __inline__ void trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.data.size(4);
    const int out_data_dim = out.size(0);

// https://blog.csdn.net/AMDS123/article/details/79541481
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = opt.background_brightness;
        }
        return;
    } else {
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        scalar_t pos[3];
        scalar_t basis_fn[25];
        maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim,
                tree.extra_data, ray.vdir, basis_fn);

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        const scalar_t d_rgb_pad = 1 + 2 * opt.rgb_padding;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
            }

            int64_t node_id;
            scalar_t* tree_val = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            scalar_t sigma = tree_val[data_dim - 1];
            if (opt.density_softplus) sigma = _SOFTPLUS_M1(sigma);
            if (sigma > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                if (opt.format != FORMAT_RGBA) {
                    for (int t = 0; t < out_data_dim; ++ t) {
                        int off = t * opt.basis_dim;
                        scalar_t tmp = 0.0;
                        for (int i = opt.min_comp; i <= opt.max_comp; ++i) {
                            tmp += basis_fn[i] * tree_val[off + i];
                        }
                        out[t] += weight * (_SIGMOID(tmp) * d_rgb_pad - opt.rgb_padding);
                    }
                } else {
                    for (int j = 0; j < out_data_dim; ++j) {
                        out[j] += weight * (_SIGMOID(tree_val[j]) * d_rgb_pad - opt.rgb_padding);
                    }
                }
                light_intensity *= att;

                if (tree.weight_accum != nullptr) {
                    if (tree.weight_accum_max) {
                        atomicMax(&tree.weight_accum[node_id], weight);
                    } else {
                        atomicAdd(&tree.weight_accum[node_id], weight);
                    }
                }

                if (light_intensity <= opt.stop_thresh) {
                    // Full opacity, stop
                    scalar_t scale = 1.0 / (1.0 - light_intensity);
                    for (int j = 0; j < out_data_dim; ++j) {
                        out[j] *= scale;
                    }
                    return;
                }
            }
            t += delta_t;
        }
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] += light_intensity * opt.background_brightness;
        }
    }
}

template <typename scalar_t>
__device__ __inline__ void cam2world_ray(
    int ix, int iy,
    scalar_t* dir,
    scalar_t* origin,
    const PackedCameraSpec<scalar_t>& __restrict__ cam) {
    scalar_t x = (ix - 0.5 * cam.width) / cam.fx;
    scalar_t y = -(iy - 0.5 * cam.height) / cam.fy;
    scalar_t z = sqrtf(x * x + y * y + 1.0);
    x /= z; y /= z; z = -1.0f / z;
    dir[0] = cam.c2w[0][0] * x + cam.c2w[0][1] * y + cam.c2w[0][2] * z;
    dir[1] = cam.c2w[1][0] * x + cam.c2w[1][1] * y + cam.c2w[1][2] * z;
    dir[2] = cam.c2w[2][0] * x + cam.c2w[2][1] * y + cam.c2w[2][2] * z;
    origin[0] = cam.c2w[0][3]; origin[1] = cam.c2w[1][3]; origin[2] = cam.c2w[2][3];
}


template <typename scalar_t>
__host__ __device__ __inline__ static void maybe_world2ndc(
        RenderOptions& __restrict__ opt,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen, scalar_t near = 1.f) {
    if (opt.ndc_width < 0)
        return;
    scalar_t t = -(near + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * opt.ndc_focal) / opt.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * opt.ndc_focal) / opt.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 * near / cen[2];

    cen[0] = -((2 * opt.ndc_focal) / opt.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * opt.ndc_focal) / opt.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 * near / cen[2];

    _normalize(dir);
}

template <typename scalar_t>
__global__ void render_image_kernel(
    PackedTreeSpec<scalar_t> tree,
    PackedCameraSpec<scalar_t> cam,
    RenderOptions opt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    maybe_world2ndc(opt, dir, origin);

    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, vdir},
        opt,
        out[iy][ix]);
}

}  // namespace device


// Compute RGB output dimension from input dimension & SH degree
__host__ int get_out_data_dim(int format, int basis_dim, int in_data_dim) {
    if (format != FORMAT_RGBA) {
        return (in_data_dim - 1) / basis_dim;
    } else {
        return in_data_dim - 1;
    }
}

}  // namespace

#define PI 3.141592653589793

using torch::Tensor;


torch::Tensor trans_t(double t) {
  return torch::tensor({{1., 0., 0., 0.}, \
                        {0., 1., 0., 0.}, \
                        {0., 0., 1., t }, \
                        {0., 0., 0., 1.}});
}

torch::Tensor rot_phi(double phi) {
  return torch::tensor({{1., 0., 0., 0.}, \
                        {0., std::cos(phi),-std::sin(phi), 0.}, \
                        {0., std::sin(phi), std::cos(phi), 0.}, \
                        {0., 0., 0., 1.}});
}

torch::Tensor rot_theta(double th) {
  return torch::tensor({{std::cos(th),0.,-std::sin(th), 0.}, \
                        {0., 1., 0., 0.}, \
                        {std::sin(th),0., std::cos(th), 0.}, \
                        {0., 0., 0., 1.}});
}

torch::Tensor pose_spherical(double theta, double phi, double radius) {
  torch::Tensor c2w = trans_t(radius);
  c2w = torch::matmul(rot_phi(phi/180.*PI), c2w);
  c2w = torch::matmul(rot_theta(theta/180.*PI), c2w);
  c2w = torch::matmul(torch::tensor({{-1., 0., 0., 0.}, {0., 0., 1., 0.}, \
                      {0., 1., 0., 0.}, {0., 0., 0., 1.}}), c2w);
  return c2w;
}


torch::Tensor volume_render_image(TreeSpec& tree, CameraSpec& cam, RenderOptions& opt) {
    tree.check();
    cam.check();
    DEVICE_GUARD(tree.data); // Set the current CUDA device to the passed Device
    const size_t Q = size_t(cam.width) * cam.height;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(opt.format, opt.basis_dim, tree.data.size(4));
    torch::Tensor result = torch::zeros({cam.height, cam.width, out_data_dim},
            tree.data.options()); // TensorOptions

    // https://zhuanlan.zhihu.com/p/48463543
    // https://pytorch.org/cppdocs/notes/tensor_basics.html
    AT_DISPATCH_FLOATING_TYPES(tree.data.type(), __FUNCTION__, [&] {
            device::render_image_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, cam, opt,
                    result.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}


int main() {
  

  torch::jit::script::Module tree_spec_dict = torch::jit::load("tree_spec_dict.pt", torch::kCUDA);
  //tree_spec_dict.to(at::kCUDA);

  TreeSpec tree_spec;
  tree_spec.data = tree_spec_dict.attr("data").toTensor();
  tree_spec.child = tree_spec_dict.attr("child").toTensor();
  tree_spec.parent_depth = tree_spec_dict.attr("parent_depth").toTensor();
  tree_spec.extra_data = tree_spec_dict.attr("extra_data").toTensor();
  tree_spec.offset = tree_spec_dict.attr("offset").toTensor();
  tree_spec.scaling = tree_spec_dict.attr("scaling").toTensor();
  tree_spec._weight_accum = tree_spec_dict.attr("_weight_accum").toTensor();
  tree_spec._weight_accum_max = tree_spec_dict.attr("_weight_accum_max").toBool();

  torch::jit::script::Module options_dict = torch::jit::load("options_dict.pt", torch::kCUDA);

  RenderOptions options;
  options.step_size = options_dict.attr("step_size").toDouble();
  options.background_brightness = options_dict.attr("background_brightness").toDouble();
  options.format = options_dict.attr("format").toInt();
  options.basis_dim = options_dict.attr("basis_dim").toInt();
  options.ndc_width = options_dict.attr("ndc_width").toInt();
  options.ndc_height = options_dict.attr("ndc_height").toInt();
  options.ndc_focal = options_dict.attr("ndc_focal").toDouble();
  options.min_comp = options_dict.attr("min_comp").toInt();
  options.max_comp = options_dict.attr("max_comp").toInt();
  options.sigma_thresh = options_dict.attr("sigma_thresh").toDouble();
  options.stop_thresh = options_dict.attr("stop_thresh").toDouble();
  options.density_softplus = options_dict.attr("density_softplus").toBool();
  options.rgb_padding = options_dict.attr("rgb_padding").toDouble();

  auto c2w = pose_spherical(90,-30,4);
  std::cout << c2w << std::endl;
  
  CameraSpec camera_spec;
  camera_spec.c2w = c2w.to(torch::device(torch::kCUDA));
  camera_spec.fx = 1111.111;
  camera_spec.fy = 1111.111;
  camera_spec.width = 800;
  camera_spec.height = 800;

  auto result = volume_render_image(tree_spec, camera_spec, options);
  
  std::cout << result.sizes() << std::endl;

  torch::save({result.to(torch::device(torch::kCPU))}, "rgb_map.pt");
}
