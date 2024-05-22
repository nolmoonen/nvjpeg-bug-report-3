#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <vector>

#define CUDA_CALL(call)                                                        \
  do {                                                                         \
    cudaError_t res_ = call;                                                   \
    if (cudaSuccess != res_) {                                                 \
      std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << "code=" << static_cast<unsigned int>(res_) << "("           \
                << cudaGetErrorString(res_) << ") \"" << #call << "\"\n";      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NVJPEG_CALL(call)                                                      \
  do {                                                                         \
    nvjpegStatus_t res_ = call;                                                \
    if (NVJPEG_STATUS_SUCCESS != res_) {                                       \
      std::cout << "nvJPEG error at " << __FILE__ << ":" << __LINE__           \
                << " code=" << static_cast<unsigned int>(res_) << "\n";        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  float truncated_fraction{};
  if (argc < 2) {
    std::cout << "usage: repro <truncated_fraction>\n";
    return EXIT_FAILURE;
  }

  std::istringstream ss(argv[1]);
  if (!(ss >> truncated_fraction)) {
    std::cout << "failed to parse\n";
    return EXIT_FAILURE;
  }

  // generate simple image

  const int size_x = 256;
  const int size_y = 256;
  const int num_components = 3;
  const int component_size = size_x * size_y;
  std::vector<std::vector<uint8_t>> h_img(3,
                                          std::vector<uint8_t>(component_size));
  for (int y = 0; y < size_y; ++y) {
    for (int x = 0; x < size_x; ++x) {
      const int idx = y * size_x + x;
      const float s = (x + 0.5f) / size_x;
      const float t = (y + 0.5f) / size_y;
      h_img[0][idx] = static_cast<uint8_t>(s * 255.0f);
      h_img[1][idx] = static_cast<uint8_t>(t * 255.0f);
      h_img[2][idx] = static_cast<uint8_t>(s * t * 255.0f);
    }
  }

  cudaStream_t stream = nullptr;

  nvjpegHandle_t nv_handle;
  NVJPEG_CALL(nvjpegCreateSimple(&nv_handle));

  // encode image

  nvjpegEncoderState_t nv_enc_state;
  NVJPEG_CALL(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));

  nvjpegEncoderParams_t nv_enc_params;
  NVJPEG_CALL(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

  NVJPEG_CALL(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params,
                                                    NVJPEG_CSS_444, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetEncoding(
      nv_enc_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetQuality(nv_enc_params, 90, stream));

  nvjpegImage_t nv_image{};
  for (int c = 0; c < num_components; ++c) {
    CUDA_CALL(cudaMalloc(&nv_image.channel[c], component_size));
    CUDA_CALL(cudaMemcpy(nv_image.channel[c], h_img[c].data(), component_size,
                         cudaMemcpyHostToDevice));
    nv_image.pitch[c] = size_x;
  }

  NVJPEG_CALL(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                &nv_image, NVJPEG_INPUT_RGB, size_x, size_y,
                                stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  size_t length{};
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, nullptr,
                                            &length, stream));
  std::vector<uint8_t> jpeg(length);
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(
      nv_handle, nv_enc_state, reinterpret_cast<unsigned char *>(jpeg.data()),
      &length, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  // write the JPEG and truncated JPEG for visualization purposes only
  {
    std::ofstream file("test.jpg", std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(jpeg.data()), length);
  }
  const size_t length_truncated = length * truncated_fraction;
  {
    std::ofstream file("test_trunc.jpg", std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(jpeg.data()), length_truncated);
  }

  // make a separate and tight allocation for the truncated image so
  //   memcheck can detect problems

  uint8_t *jpeg_truncated = static_cast<uint8_t *>(malloc(length_truncated));
  std::copy(jpeg.begin(), jpeg.begin() + length_truncated, jpeg_truncated);

  NVJPEG_CALL(nvjpegEncoderParamsDestroy(nv_enc_params));
  NVJPEG_CALL(nvjpegEncoderStateDestroy(nv_enc_state));

  // decode truncated image

  nvjpegJpegState_t jpeg_state{};
  NVJPEG_CALL(nvjpegJpegStateCreate(nv_handle, &jpeg_state));

  nvjpegJpegDecoder_t nvjpeg_decoder{};
  // using the GPU_HYBRID backend seems crucial to reproducing the bug
  NVJPEG_CALL(nvjpegDecoderCreate(nv_handle, NVJPEG_BACKEND_GPU_HYBRID,
                                  &nvjpeg_decoder));
  nvjpegJpegState_t nvjpeg_decoupled_state;
  NVJPEG_CALL(nvjpegDecoderStateCreate(nv_handle, nvjpeg_decoder,
                                       &nvjpeg_decoupled_state));

  nvjpegBufferPinned_t pinned_buffer;
  NVJPEG_CALL(nvjpegBufferPinnedCreate(nv_handle, nullptr, &pinned_buffer));
  nvjpegBufferDevice_t device_buffer;
  NVJPEG_CALL(nvjpegBufferDeviceCreate(nv_handle, nullptr, &device_buffer));

  nvjpegJpegStream_t jpeg_stream;
  NVJPEG_CALL(nvjpegJpegStreamCreate(nv_handle, &jpeg_stream));

  nvjpegDecodeParams_t nvjpeg_decode_params;
  NVJPEG_CALL(nvjpegDecodeParamsCreate(nv_handle, &nvjpeg_decode_params));

  NVJPEG_CALL(
      nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
  NVJPEG_CALL(
      nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
  NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params,
                                                NVJPEG_OUTPUT_RGB));

  NVJPEG_CALL(nvjpegJpegStreamParse(
      nv_handle, reinterpret_cast<unsigned char *>(jpeg_truncated),
      length_truncated, 0, 0, jpeg_stream));

  NVJPEG_CALL(nvjpegDecodeJpegHost(nv_handle, nvjpeg_decoder,
                                   nvjpeg_decoupled_state, nvjpeg_decode_params,
                                   jpeg_stream));

  NVJPEG_CALL(nvjpegDecodeJpegTransferToDevice(
      nv_handle, nvjpeg_decoder, nvjpeg_decoupled_state, jpeg_stream, stream));

  NVJPEG_CALL(nvjpegDecodeJpegDevice(
      nv_handle, nvjpeg_decoder, nvjpeg_decoupled_state, &nv_image, stream));

  CUDA_CALL(cudaStreamSynchronize(stream));

  NVJPEG_CALL(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
  NVJPEG_CALL(nvjpegJpegStreamDestroy(jpeg_stream));
  NVJPEG_CALL(nvjpegBufferPinnedDestroy(pinned_buffer));
  NVJPEG_CALL(nvjpegBufferDeviceDestroy(device_buffer));
  NVJPEG_CALL(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
  NVJPEG_CALL(nvjpegDecoderDestroy(nvjpeg_decoder));
  NVJPEG_CALL(nvjpegJpegStateDestroy(jpeg_state));

  // if the decoding is successful, output the image

  for (int c = 0; c < num_components; ++c) {
    CUDA_CALL(cudaMemcpy(h_img[c].data(), nv_image.channel[c], component_size,
                         cudaMemcpyDeviceToHost));
  }

  {
    std::ofstream file("out.ppm", std::ios::out | std::ios::binary);
    file << "P6\n" << size_x << " " << size_y << "\n255\n";
    for (int i = 0; i < size_x * size_y; ++i) {
      file << h_img[0][i] << h_img[1][i] << h_img[2][i];
    }
  }

  free(jpeg_truncated);

  for (int c = 0; c < num_components; ++c) {
    CUDA_CALL(cudaFree(nv_image.channel[c]));
  }
}
