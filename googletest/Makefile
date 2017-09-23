#!/bin/bash -e

BIN_DIR=../bin/googletest

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

downloaded_deps/googletest.tar.gz :
	mkdir -p downloaded_deps
	cd downloaded_deps && \
	curl -o googletest.tar.gz https://codeload.github.com/google/googletest/tar.gz/release-1.8.0

downloaded_deps/googletest : downloaded_deps/googletest.tar.gz
	cd downloaded_deps && \
	tar xf googletest.tar.gz && \
	ln -s googletest-release-1.8.0 googletest && \
	cd googletest/googletest && \
	cmake .

clean:
	rm -Rf downloaded_deps
	rm -Rf $(BIN_DIR)

CXX = clang++
GTEST_DIR = downloaded_deps/googletest/googletest/

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wall -Wextra -pthread

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
$(BIN_DIR)/gtest-all.o : $(BIN_DIR) downloaded_deps/googletest $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc \
			-o $@

$(BIN_DIR)/gtest_main.o : $(BIN_DIR) downloaded_deps/googletest $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc \
			-o $@

$(BIN_DIR)/gtest.a : $(BIN_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(BIN_DIR)/gtest_main.a : $(BIN_DIR)/gtest-all.o $(BIN_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

