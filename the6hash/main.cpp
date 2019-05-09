//
//  main.cpp
//  the6hash
//
//  Created by 朱瑞 on 19/5/9.
//  Copyright © 2019年 朱瑞. All rights reserved.
//

#include <iostream>
#include "hashsix.hpp"
int main() {
    // insert code here...
    std::cout << "Hello, World!\n"<<std::endl;
     cv::Mat img1=cv::imread("pic.png",CV_LOAD_IMAGE_GRAYSCALE);
    Hash hash;
    std::cout<<hash.average_hash(img1, 36, 64)<<std::endl;
//    std::cout<<hash.block_hash(img1, 36, 64)<<std::endl;
//    std::cout<<hash.difference_hash(img1, 36, 64)<<std::endl;
//    std::cout<<hash.median_hash(img1, 36, 64)<<std::endl;
//    std::cout<<hash.perceptual_hash(img1, 36, 64)<<std::endl;
//    std::cout<<hash.wavelet_hash(img1, 36, 64)<<std::endl;
    return 0;
}
