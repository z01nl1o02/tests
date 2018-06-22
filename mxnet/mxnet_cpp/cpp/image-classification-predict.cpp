/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 *
 * This is a simple predictor which shows how to use c api for image classification. It uses
 * opencv for image reading.
 *
 * Created by liuxiao on 12/9/15.
 * Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 * Home Page: www.liuxiao.org
 * E-mail: liuxiao@foxmail.com
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>
#include <opencv2/opencv.hpp>
// Path for c_predict_api
#include "mxnet/c_predict_api.h"
#include "iniio.h"



class RUNPARAM
{
private:
	std::string weightsFilePath_, netFilePath_;
	std::string labelFilePath_;
	std::string meanFilePath_;
	std::string imageListPath_;
	int width_, height_, channelNum_;

	float meanB_, meanG_, meanR_;

	std::string device_;
public:
	bool load(const char* filepath)
	{
		iniio::IO io;
		int ret = io.load(filepath);
		assert(ret == iniio::OK);

		std::string root = io.getStringValue("req", "root", "");

		weightsFilePath_ = root + io.getStringValue("req", "weights", "net.params");
		netFilePath_ = root + io.getStringValue("req", "net", "net.json");
		imageListPath_ = root + io.getStringValue("req", "image", "images.txt");

		width_ = io.getIntValue("req", "width", 28);
		height_ = io.getIntValue("req", "height", 28);
		channelNum_ = io.getIntValue("req", "channel", 3);

		labelFilePath_ = io.getStringValue("opt", "label", "");
		if (labelFilePath_.empty() == false) labelFilePath_ = root + labelFilePath_;
		meanFilePath_ = io.getStringValue("opt", "mean", "");
		if (meanFilePath_.empty() == false) meanFilePath_ = root + meanFilePath_;

	
		device_ = io.getStringValue("opt", "device", "cpu");

		meanB_ = io.getFloatValue("opt", "meanb", 0);
		meanG_ = io.getFloatValue("opt", "meang", 0);
		meanR_ = io.getFloatValue("opt", "meanr", 0);
		return true;
	}
public:
	RUNPARAM()
	{
	}
	~RUNPARAM()
	{
		return;
	}
public:
	std::string weightFile() { return weightsFilePath_;  }
	std::string netFile() { return netFilePath_;  }
	std::string labelFile() { return labelFilePath_; }
	std::string meanFile() { return meanFilePath_; }
	std::string imageListFile() { return imageListPath_;}
	int width() { return width_;  }
	int height() { return height_;  }
	int channel() { return channelNum_;  }

	int meanB() { return meanB_; }
	int meanG() { return meanG_; }
	int meanR() { return meanR_; }

	std::string device() { return device_; }

};


RUNPARAM g_param;

static std::string trim(const std::string& input) {
  auto not_space = [](int ch) {
    return !std::isspace(ch);
  };
  auto output = input;
  output.erase(output.begin(), std::find_if(output.begin(), output.end(), not_space));
  output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(), output.end());
  return output;
}

// Read file to buffer
class BufferFile {
 public :
  std::string file_path_;
  std::size_t length_ = 0;
  std::unique_ptr<char[]> buffer_;

  explicit BufferFile(const std::string& file_path)
    : file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_.reset(new char[length_]);
    ifs.read(buffer_.get(), length_);
    ifs.close();
  }

  std::size_t GetLength() {
    return length_;
  }

  char* GetBuffer() {
    return buffer_.get();
  }
};

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size, const mx_float* mean_data = nullptr) {

#if 1
  // Read all kinds of file into a BGR color 3 channels image
  cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

  if (im_ori.empty()) {
    std::cerr << "Can't open the image. Please check " << image_file << ". \n";
    assert(false);
  }

  cv::Mat im;

  resize(im_ori, im, resize_size);

  int size = im.rows * im.cols * channels;

  mx_float* ptr_image_r = image_data;
  mx_float* ptr_image_g = image_data + size / 3;
  mx_float* ptr_image_b = image_data + size / 3 * 2;

  float mean_b, mean_g, mean_r;
  mean_b = g_param.meanB();
  mean_g = g_param.meanG();
  mean_r = g_param.meanR();

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
      if (mean_data) {
        mean_r = *mean_data;
        if (channels > 1) {
          mean_g = *(mean_data + size / 3);
          mean_b = *(mean_data + size / 3 * 2);
        }
        mean_data++;
      }
      if (channels > 1) {
        *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
        *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
      }

      *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
    }
  }


#else
  // Read all kinds of file into a BGR color 3 channels image
  cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

  if (im_ori.empty()) {
    std::cerr << "Can't open the image. Please check " << image_file << ". \n";
    assert(false);
  }

  cv::Mat im;

  resize(im_ori, im, resize_size);

  int size = im.rows * im.cols * channels;

  mx_float* ptr_image_r = image_data;
  mx_float* ptr_image_g = image_data + size / 3;
  mx_float* ptr_image_b = image_data + size / 3 * 2;

  float mean_b, mean_g, mean_r;
  mean_b = mean_g = mean_r = DEFAULT_MEAN;

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
      if (mean_data) {
        mean_r = *mean_data;
        if (channels > 1) {
          mean_g = *(mean_data + size / 3);
          mean_b = *(mean_data + size / 3 * 2);
        }
        mean_data++;
      }
      if (channels > 1) {
        *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
        *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
      }

      *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
    }
  }

#endif
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(const std::string& synset_file) {
  std::ifstream fi(synset_file.c_str());

  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    assert(false);
  }

  std::vector<std::string> output;

  std::string synset, lemma;
  while (fi >> synset) {
    getline(fi, lemma);
    output.push_back(lemma);
  }

  fi.close();

  return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
  if (data.size() != synset.size()) {
    std::cerr << "Result data and synset size do not match!" << std::endl;
  }

  float best_accuracy = 0.0;
  std::size_t best_idx = 0;

  for (std::size_t i = 0; i < data.size(); ++i) {
    //std::cout << "Accuracy[" << i << "] = " << std::setprecision(8) << data[i] << std::endl;

    if (data[i] > best_accuracy) {
      best_accuracy = data[i];
      best_idx = i;
    }
  }

  std::cout << "Best Result: " << trim(synset[best_idx]) << " (id=" << best_idx << ", " <<
            "accuracy=" << std::setprecision(8) << best_accuracy << ")" << std::endl;
}

std::vector<std::string> load_filelist(const char* filepath)
{
	std::ifstream fi(filepath);

	if (!fi.is_open()) {
		std::cerr << "Error opening synset file " << filepath << std::endl;
		assert(false);
	}

	std::vector<std::string> output;

	std::string path;
	while (getline(fi,path) )
	{
		output.push_back(path);
	}
	fi.close();
	return output;

}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "No test image here." << std::endl
              << "Usage: ./image-classification-predict run.ini" << std::endl;
    return EXIT_FAILURE;
  }

  //std::string test_file("guitar.jpg");
  // Models path for your model, you have to modify it
  /*
  std::string json_file = "model/Inception/Inception-BN-symbol.json";
  std::string param_file = "model/Inception/Inception-BN-0126.params";
  std::string synset_file = "model/Inception/synset.txt";
  std::string nd_file = "model/Inception/mean_224.nd";
  */

  if (false == g_param.load(argv[1]))
  {
	  std::cout << "load " << argv[1] << " failed" << std::endl;
	  return EXIT_FAILURE;
  }

  std::string json_file = g_param.netFile();
  std::string param_file = g_param.weightFile();
  std::string synset_file = g_param.labelFile();
  std::string nd_file = g_param.meanFile();

  BufferFile json_data(json_file);
  BufferFile param_data(param_file);

  // Parameters
  int dev_type = g_param.device() == "cpu" ? 1 : 2;
  int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" };
  const char** input_keys = input_key;

  // Image size and channels
  //int width = 224;
  //int height = 224;
  int width = g_param.width();
  int height = g_param.height();
  int channels = g_param.channel();

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height),
                                        static_cast<mx_uint>(width) };
  PredictorHandle pred_hnd = nullptr;

  if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
    return EXIT_FAILURE;
  }

  // Create Predictor
  MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
               static_cast<const char*>(param_data.GetBuffer()),
               static_cast<int>(param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);
  assert(pred_hnd);

  auto image_size = static_cast<std::size_t>(width * height * channels);

  // Read Mean Data
  const mx_float* nd_data = nullptr;
  NDListHandle nd_hnd = nullptr;
  
  if (nd_file != "")
  {
	  BufferFile nd_buf(nd_file);
	  if (nd_buf.GetLength() > 0) {
		  mx_uint nd_index = 0;
		  mx_uint nd_len;
		  const mx_uint* nd_shape = nullptr;
		  const char* nd_key = nullptr;
		  mx_uint nd_ndim = 0;

		  MXNDListCreate(static_cast<const char*>(nd_buf.GetBuffer()),
			  static_cast<int>(nd_buf.GetLength()),
			  &nd_hnd, &nd_len);

		  MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
	  }
  }
// Synset path for your model, you have to modify it
  auto synset = LoadSynset(g_param.labelFile());
  std::vector< std::string > test_files = load_filelist(g_param.imageListFile().c_str());
  for(int fidx = 0; fidx < test_files.size(); fidx++)
  {
	  std::string test_file = test_files[fidx];
	  // Read Image Data
	  std::vector<mx_float> image_data(image_size);

	  GetImageFile(test_file, image_data.data(), channels, cv::Size(width, height), nd_data);

	  // Set Input Image
	  MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

	  // Do Predict Forward
	  MXPredForward(pred_hnd);

	  mx_uint output_index = 0;

	  mx_uint* shape = nullptr;
	  mx_uint shape_len;

	  // Get Output Result
	  MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

	  std::size_t size = 1;
	  for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

	  std::vector<float> data(size);

	  MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

	  // Print Output Data
	  PrintOutputResult(data, synset);
  }


  // Release NDList
  if (nd_hnd) {
    MXNDListFree(nd_hnd);
  }

  // Release Predictor
  MXPredFree(pred_hnd);

  return EXIT_SUCCESS;
}
