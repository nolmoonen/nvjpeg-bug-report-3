# nvjpeg-bug-report-3

Demonstrates `nvjpegDecodeJpegHost` reading outside the bit-stream allocation. The reproducer encodes a simple image and decodes a truncated part of the encoded JPEG stream.

Tested with CUDA 12.2 driver version 535.113.01 on Ubuntu 22.04.1.

Not to be confused with https://github.com/nolmoonen/nvjpeg-bug-report or https://github.com/nolmoonen/nvjpeg-bug-report-2.

## Example
Running `./repro 0.9` will decode all but the last 10% of encoded bytes. The full encoded test image is [test.jpg](test.jpg). The truncated image is [test_trunc.jpg](test_trunc.jpg). Output of `valgrind --tool=memcheck ./repro 0.9` can be found at [out.txt](out.txt), important snippet:

```shell
==94834== Invalid read of size 1
==94834==    at 0x2BAA7D: nvjpeg::DecodeSingleGPU::nppiDecodeHuffmanSweepHost(nvjpeg::DecodeSingleGPU::NppiJpegDecodeJob const&, nvjpeg::DecodeSingleGPU::NppiJpegDecodeJobMemory const&) (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
==94834==    by 0x1DEBBF: nvjpeg::DecodeSingleGPU::CodecJPEGHuffmanGPU::callHuffmanSweep(nvjpeg::DecodeSingleGPU::JpegGPUSingleImageState<nvjpeg::DecodeSingleGPU::CodecJPEGHuffmanGPU>&, nvjpeg::DecodeSingleGPU::NppiJpegFrameDescr*, nvjpeg::FrameHeader&, nvjpeg::ParsedJpeg const&) (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
==94834==    by 0x1DF529: nvjpeg::DecodeSingleGPU::CodecJPEGHuffmanGPU::decodeHost(nvjpeg::IDecoderState*, nvjpeg::DecodeParams const&, nvjpeg::ParsedJpeg const&) (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
==94834==    by 0x1A9669: nvjpegDecodeJpegHost (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
==94834==    by 0x11B874: main (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
==94834==  Address 0xd630ba1 is 0 bytes after a block of size 6,625 alloc'd
==94834==    at 0x484880F: malloc (vg_replace_malloc.c:431)
==94834==    by 0x11ADDC: main (in /home/nol/dev/nvjpeg-bug-report-3/build/repro)
```

It is not expected that the image is successfully decoded (the reproducer decodes a non-standard JPEG), but it is expected that no memory out of bounds is accessed.
