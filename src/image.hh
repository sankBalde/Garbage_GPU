#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>
#include <cstdint>

struct Image
{
    Image() = default;

    Image(const std::string &filepath, int id = -1)
    {
        to_sort.id = id;

        std::ifstream infile(filepath, std::ifstream::binary);

        if (!infile.is_open())
            throw std::runtime_error("Failed to open");

        std::string magic;
        infile >> magic;
        infile.seekg(1, infile.cur);
        char c;
        infile.get(c);
        while (c == '#')
        {
            while (c != '\n')
                infile.get(c);
            infile.get(c);
        }

        infile.seekg(-1, infile.cur);

        int max;
        infile >> width >> height >> max;
        if (max != 255 && magic == "P5")
            throw std::runtime_error("Bad max value");

        if (magic == "P5")
        {
            // TODO : Isn't there a better way to allocate the CPU Memory
            // To speed up the Host-to-Device Transfert ?
            buffer = (int *)malloc(width * height * sizeof(int));
            infile.seekg(1, infile.cur);
            for (int i = 0; i < width * height; ++i)
            {
                uint8_t pixel_char;
                infile >> std::noskipws >> pixel_char;
                buffer[i] = pixel_char;
            }
            actual_size = width * height;
        }
        else if (magic == "P?")
        {
            infile.seekg(1, infile.cur);

            std::string line;
            std::getline(infile, line);

            int image_size = 0;
            {
                std::stringstream lineStream(line);
                std::string s;

                while (std::getline(lineStream, s, ';'))
                    ++image_size;
            }
            // TODO : Isn't there a better way to allocate the CPU Memory
            // To speed up the Host-to-Device Transfert ?
            buffer = (int *)malloc(image_size * sizeof(int));

            std::stringstream lineStream(line);
            std::string s;

            int i = 0;

            while (std::getline(lineStream, s, ';'))
                buffer[i++] = std::stoi(s);
            actual_size = i;
        }
        else
            throw std::runtime_error("Bad PPM value");
    }

    int size() const
    {
        return actual_size;
    }

    void write(const std::string &filepath) const
    {
        std::ofstream outfile(filepath, std::ofstream::binary);
        if (outfile.fail())
            throw std::runtime_error("Failed to open");
        outfile << "P5" << "\n"
                << width << " " << height << "\n"
                << 255 << "\n";

        for (int i = 0; i < height * width; ++i)
        {
            int val = buffer[i];
            if (val < 0 || val > 255)
            {
                std::cout << std::endl;
                std::cout << "Error at : " << i << " Value is : " << val << ". Values should be between 0 and 255." << std::endl;
                throw std::runtime_error("Invalid image format");
            }
            outfile << static_cast<uint8_t>(val);
        }
    }

    int *buffer;
    int height = -1;
    int width = -1;
    int actual_size = -1;
    struct ToSort
    {
        uint64_t total = 0;
        int id = -1;
    } to_sort;
};