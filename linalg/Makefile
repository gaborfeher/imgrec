NVCC=/usr/local/cuda-8.0/bin/nvcc

hello: hello.cpp DeviceMatrix.cu HostMatrix.cpp DeviceMatrix.h HostMatrix.h
	$(NVCC) hello.cpp DeviceMatrix.cu HostMatrix.cpp \
		--output-file=hello -Wno-deprecated-gpu-targets \
		--compiler-bindir=clang++ \
		--std=c++11

clean:
	rm -f ./hello
