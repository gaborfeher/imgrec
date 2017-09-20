NVCC=/usr/local/cuda-8.0/bin/nvcc

hello: hello.cu
	$(NVCC) hello.cu --output-file=hello -Wno-deprecated-gpu-targets

clean:
	rm -f ./hello
