# Makefile for MetaMind Examples
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -I..
TARGET = metamind_demo
SRC = metamind_demo.cpp

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRC) ../metamind.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "MetaMind Examples Makefile"
	@echo "Targets: all, run, clean, help"
