# Makefile for tensor project

# Directories
SRC_DIR := src
INC_DIR := include
TEST_DIR := tests
BUILD_DIR := build

# Compiler and flags
CC := clang
CFLAGS := -Wall -Wextra -g -I$(INC_DIR)
LDFLAGS :=

# Source files
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
TEST_FILES := $(wildcard $(TEST_DIR)/*_test.c)

# Object files - separate directories for src and test objects
SRC_OBJ_DIR := $(BUILD_DIR)/src_obj
TEST_OBJ_DIR := $(BUILD_DIR)/test_obj

SRC_OBJS := $(patsubst $(SRC_DIR)/%.c,$(SRC_OBJ_DIR)/%.o,$(SRC_FILES))
TEST_OBJS := $(patsubst $(TEST_DIR)/%_test.c,$(TEST_OBJ_DIR)/%.o,$(TEST_FILES))
OBJ_FILES := $(SRC_OBJS) $(TEST_OBJS)

# Target executable
TARGET := $(BUILD_DIR)/test_tensor

# Default target
all: build

build: directories $(TARGET)

directories:
	@mkdir -p $(BUILD_DIR) $(SRC_OBJ_DIR) $(TEST_OBJ_DIR)

$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Compile source files
$(SRC_OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test files
$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%_test.c
	$(CC) $(CFLAGS) -c $< -o $@

test: build
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all build test clean directories