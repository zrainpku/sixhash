//
//  hashsix.hpp
//  the6hash
//
//  Created by 朱瑞 on 19/5/9.
//  Copyright © 2019年 朱瑞. All rights reserved.
//

#ifndef hashsix_hpp
#define hashsix_hpp

#include <stdio.h>
#include<opencv2/opencv.hpp>

class WaveTransform{
private:
    void wavelet( const std::string _wname, cv::Mat &_lowFilter, cv::Mat &_highFilter );
    cv::Mat waveletDecompose( const cv::Mat &_src, const cv::Mat &_lowFilter, const cv::Mat &_highFilter );
    
public:
    cv::Mat WDT(const cv::Mat &_src,const std::string _wname,const int _level);
    
};


class Hash{
public:
    std::string outlinehash_row(cv::Mat src,int num);
    std::string outlinehash_col(cv::Mat src,int num);
    
    std::string average_hash(cv::Mat &img,int r,int c);
    std::string block_hash(cv::Mat &img,int r,int c);
    std::string outline_hash(cv::Mat &img,int r,int c);
    std::string difference_hash(cv::Mat &img,int r,int c);
    std::string median_hash(cv::Mat &img,int r,int c);
    std::string perceptual_hash(cv::Mat &img,int r,int c);
    std::string wavelet_hash(cv::Mat &img,int r,int c);
    
    float HanmingDistance(std::string &str1,std::string &str2,int len);
    float OutlineDistance(std::string &str1,std::string &str2,int len);
    

    std::string get_average_hash(cv::Mat &img,int r,int c);
    std::string get_outline_hash(cv::Mat &img,int r,int c);
    
    
    float Ans_average_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_block_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_difference_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_outline_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_median_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_perceptual_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
    float Ans_wavelet_hash(cv::Mat &img1,cv::Mat &img2,int r,int c);
};




#endif /* hashsix_hpp */
