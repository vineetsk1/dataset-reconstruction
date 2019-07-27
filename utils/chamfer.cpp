#include <torch/torch.h>

void nnsearch(const int b, const int n, const int m,
              const float* xyz1, const float* xyz2,
              float* dist, int* idx);

int CDKernelLauncher(const int b, const int n,
                     const float* xyz1, const int m,
                     const float* xyz2, float* dist1,
                     int* idx1, float* dist2, int* idx2);

// Chamfer Distance
// Forward implementation
// CPU
void cd_forward(const at::Tensor xyz1,
                const at::Tensor xyz2,
                const at::Tensor dist1,
                const at::Tensor dist2,
                const at::Tensor idx1,
                const at::Tensor idx2) {

    const int batchsize = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();

    float* dist1_data = dist1.data<float>();
    float* dist2_data = dist2.data<float>();
    int* idx1_data = idx1.data<int>();
    int* idx2_data = idx2.data<int>();

    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);
}

// Chamfer Distance
// Forward implementation
// CPU
void nnsearch(const int b, const int n, const int m,
              const float* xyz1, const float* xyz2,
              float* dist, int* idx) {
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1[(i*n+j)*3+0];
            const float y1 = xyz1[(i*n+j)*3+1];
            const float z1 = xyz1[(i*n+j)*3+2];

            double best = 0;
            int besti = 0;
            for (int k = 0; k < m; k++) {
                const float x2 = xyz2[(i*m+k)*3+0] - x1;
                const float y2 = xyz2[(i*m+k)*3+1] - y1;
                const float z2 = xyz2[(i*m+k)*3+2] - z1;
                const double d = x2*x2 + y2*y2 + z2*z2;
                if (k == 0 || d < best) {
                    best = d;
                    besti = k;
                }
            }
            dist[i*n+j] = best;
            idx[i*n+j] = besti;
        }
    }
}

// Chamfer Distance
// Forward implementation
// CUDA
void cd_forward_cuda(const at::Tensor xyz1,
                     const at::Tensor xyz2,
                     const at::Tensor dist1,
                     const at::Tensor dist2,
                     const at::Tensor idx1,
                     const at::Tensor idx2) {
    CDKernelLauncher(xyz1.size(0), xyz1.size(1), xyz1.data<float>(),
                     xyz2.size(1), xyz2.data<float>(), dist2.data<float>(),
                     idx1.data<int>(), dist2.data<float>(), idx2.data<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cd_forward, "CD forward");
    m.def("forward_cuda", &cd_forward_cuda, "CD forward CUDA");
}