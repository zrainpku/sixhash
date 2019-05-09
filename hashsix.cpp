//
//  hashsix.cpp
//  the6hash
//
//  Created by 朱瑞 on 19/5/9.
//  Copyright © 2019年 朱瑞. All rights reserved.
//

#include "hashsix.hpp"

#include <math.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
cv::Mat WaveTransform::WDT(const cv::Mat &_src,const std::string _wname,const int _level)
{
    cv::Mat src=cv::Mat_<float>(_src);
    cv::Mat dst=cv::Mat::zeros(src.rows,src.cols,src.type());
    int N=src.rows;
    int D=src.cols;
    //高通低通滤波器
    cv::Mat lowFilter;
    cv::Mat highFilter;
    wavelet(_wname,lowFilter,highFilter);
    //小波变换
    int t=1;
    int row=N;
    int col=D;
    while (t<=_level)
    {
        //先进行 行小波变换
        for (int i=0;i<row;i++)
        {
            //取出src中要处理的数据的一行
            cv::Mat oneRow=cv::Mat::zeros(1,col,src.type());
            for (int j=0;j<col;j++)
            {
                oneRow.at<float>(0,j)=src.at<float>(i,j);
            }
            oneRow=waveletDecompose(oneRow,lowFilter,highFilter);
            for (int j=0;j<col;j++)
            {
                dst.at<float>(i,j)=oneRow.at<float>(0,j);
            }
        }
        //        char s[10];
        //        std::itoa(t,s,10);
        //        cv::imshow(s,dst);
        //        waitKey();
        //#if 0
        //        //    normalize(dst,dst,0,255,NORM_MINMAX);
        //        IplImage dstImg1=IplImage(dst);
        //        cvSaveImage("dst1.jpg",&dstImg1);
        //#endif
        //        cv::normalize(dst,dst,0,255,cv::NORM_MINMAX);
        //        std::string path="/Users/zrain/Desktop/scshot/hash/wavelet_hang"+std::to_string(t)+".png";
        //        cv::imwrite(path, dst);
        //小波列变换
        for (int j=0;j<col;j++)
        {
            cv::Mat oneCol=cv::Mat::zeros(row,1,src.type());
            for (int i=0;i<row;i++)
            {
                oneCol.at<float>(i,0)=dst.at<float>(i,j);//dst,not src
            }
            oneCol=(waveletDecompose(oneCol.t(),lowFilter,highFilter)).t();
            for (int i=0;i<row;i++)
            {
                dst.at<float>(i,j)=oneCol.at<float>(i,0);
            }
        }
        //#if 0
        //        //    normalize(dst,dst,0,255,NORM_MINMAX);
        //        IplImage dstImg2=IplImage(dst);
        //        cvSaveImage("dst2.jpg",&dstImg2);
        //#endif
        //        cv::normalize(dst,dst,0,255,cv::NORM_MINMAX);
        //        path="/Users/zrain/Desktop/scshot/hash/wavelet_line"+std::to_string(t)+".png";
        //
        //        cv::imwrite(path, dst);
        
        //更新
        row/=2;
        col/=2;
        src=dst;
        if(t==1)cv::imwrite("waveletA.png", src);
        if(t==2)cv::imwrite("waveletB.png", src);
        if(t==3)cv::imwrite("waveletC.png", src);
         t++;
        
    }
    return dst;
}

//生成不同类型的小波
void WaveTransform::wavelet( const std::string _wname, cv::Mat &_lowFilter, cv::Mat &_highFilter )
{
    
    if (_wname=="haar" || _wname=="db1")
    {
        int N=2;
        _lowFilter=cv::Mat::zeros(1,N,CV_32F);
        _highFilter=cv::Mat::zeros(1,N,CV_32F);
        
        _lowFilter.at<float>(0,0)=1/sqrtf(N);
        _lowFilter.at<float>(0,1)=1/sqrtf(N);
        
        _highFilter.at<float>(0,0)=-1/sqrtf(N);
        _highFilter.at<float>(0,1)=1/sqrtf(N);
    }
    if (_wname=="sym2")
    {
        int N=4;
        float h[]={-0.483, 0.836, -0.224, -0.129};
        float l[]={-0.129, 0.224,    0.837, 0.483};
        
        _lowFilter=cv::Mat::zeros(1,N,CV_32F);
        _highFilter=cv::Mat::zeros(1,N,CV_32F);
        
        for (int i=0;i<N;i++)
        {
            _lowFilter.at<float>(0,i)=l[i];
            _highFilter.at<float>(0,i)=h[i];
        }
    }
    
}

//小波分解
cv::Mat WaveTransform::waveletDecompose( const cv::Mat &_src, const cv::Mat &_lowFilter, const cv::Mat &_highFilter )
{
    assert(_src.rows==1 && _lowFilter.rows==1 && _highFilter.rows ==1);
    assert(_src.cols>=_lowFilter.cols && _src.cols>=_highFilter.cols );
    cv::Mat src=cv::Mat_<float>(_src);
    
    int D=src.cols;
    
    cv::Mat lowFilter=cv::Mat_<float>(_lowFilter);
    cv::Mat highFilter=cv::Mat_<float>(_highFilter);
    
    //频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter)
    cv::Mat dst1=cv::Mat::zeros(1,D,src.type());
    cv::Mat dst2=cv::Mat::zeros(1,D,src.type());
    
    cv::filter2D(src,dst1,-1,lowFilter);
    cv::filter2D(src,dst2,-1,highFilter);
    
    //下采样
    cv::Mat downDst1=cv::Mat::zeros(1,D/2,src.type());
    cv::Mat downDst2=cv::Mat::zeros(1,D/2,src.type());
    
    cv::resize(dst1,downDst1,downDst1.size());
    cv::resize(dst2,downDst2,downDst2.size());
    
    //数据拼接
    for (int i=0;i<D/2;i++)
    {
        src.at<float>(0,i)=downDst1.at<float>(0,i)/1.5;
        src.at<float>(0,i+D/2)=downDst2.at<float>(0,i)/1.5;
        
    }
    return src;
}

//#####################################

float Hash::HanmingDistance(std::string &str1,std::string &str2,int len)
{
    if((str1.size()!=len)||(str2.size()!=len))
        return -1;
    float difference = 0;
    for(int i=0;i<len;i++)
    {
        if(str1[i]==str2[i])
            difference++;
    }
    return difference/len;
}

float myhash(std::string str1,std::string str2,int pos){
    float res=0;
    if(str1.size()!=str2.size())
        return -1;
    std::vector<int> v1,v2;
    int len=str1.size();
    if(str1==str2){
        res=1;
    }else{
        for(int i=0;i<str1.size();i++){
            if(str1[i]=='1'){
                v1.push_back(i);
            }
            if(str2[i]=='1'){
                v2.push_back(i);
            }
        }
        if(v1.size()==0 || v2.size()==0){
            res=1.0/(v1.size()+v2.size()+1);
        }else{
            float distance=0;
            for(int i=0;i<v1.size();i++){
                float min_dis=30;
                for(int j=0;j<v2.size();j++)
                {
                    //                    if(v1[i]==v2[j]){
                    //                        distance++;
                    //                    }else if (abs(v1[i]-v2[j])==1){
                    //                        distance+=0.5;
                    //                    }
                    float t=abs(v1[i]-v2[j]);
                    min_dis=MIN(min_dis,t);
                    //                    if (abs(v1[i]-v2[j])==0){
                    //                        distance+=1;
                    //                        break;
                    //                    }
                    
                }
                if(min_dis>=4){
                    //
                    distance-=0.5;
                }else if(min_dis>=2){
                    distance-=0.2;
                }else if(min_dis>=1){
                    distance+=0.1;
                }else{
                    distance+=1;
                }
            }
            int mm=MAX(v1.size(), v2.size());
            res=(float)distance/mm;
        }
    }//else
    //    std::cout<<pos<<": "<<std::endl<<str1<<std::endl<<str2<<std::endl;
    
    if(pos<6) {
        //        std::cout<<pos<<": "<<res<<std::endl;
        return res*0.1;
    }else{
        //        std::cout<<pos<<": "<<res<<std::endl;
        return res*0.05;
    }
}

float Hash::OutlineDistance(std::string &str1,std::string &str2,int len){
    if((str1.size()!=len)||(str2.size()!=len))
        return -1;
    float difference = 0,res=0.0;
    //    std::vector<int> ls={36,64,64,64,36,36,64,64,64,64,36,36,36,36};
    std::vector<int> ls;
    ls.push_back(36);
    ls.push_back(64);
    ls.push_back(64);
    ls.push_back(64);
    ls.push_back(36);
    ls.push_back(36);
    ls.push_back(64);
    ls.push_back(64);
    ls.push_back(64);
    ls.push_back(64);
    ls.push_back(36);
    ls.push_back(36);
    ls.push_back(36);
    ls.push_back(36);
    
    int id=0;
    for(int i=0;i<ls.size();i++)
    {
        int temp=ls[i];
        float ft=0;
        if(temp==36){
            std::string ss1=str1.substr(id,36);
            std::string ss2=str2.substr(id,36);
            id+=36;
            ft=myhash(ss1,ss2,i);
        }else if (temp==64){
            std::string ss1=str1.substr(id,64);
            std::string ss2=str2.substr(id,64);
            id+=64;
            ft=myhash(ss1,ss2,i);
        }else{
            std::cout<<"ERROR"<<std::endl;
        }
        res+=ft;
        
        
    }
    
    return MAX(res, 0.0);
}



std::string Hash::average_hash(cv::Mat &img,int r,int c){
    std::string res;
    cv::Mat src=img.clone();
    cv::imwrite("average1.png", src);
    //    cv::cvtColor(src, src, CV_BGR2GRAY);
    //    cv::threshold(src, src, 40, 255,CV_THRESH_BINARY );
    //    cv::dilate(src, src, getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    cv::resize(src, src, cv::Size(c, r), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::imwrite("average2.png", src);
    int mean=0;
    for(int i=0;i<r;i++ ){
        for(int j=0;j<c;j++){
            mean+=src.at<uchar>(i,j);
        }
    }
    mean/=r*c;
    //    std::cout<<"mean: "<<mean<<std::endl;
    cv::Mat dest(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++ ){
        for(int j=0;j<c;j++){
            //            std::cout<<"i: "<<i<<"j: "<<j<<(int)src.at<uchar>(i,j)<<std::endl;
            if((int)src.at<uchar>(i,j)>mean){
                dest.at<uchar>(i,j)=255;
                res+="1";
            }else{
                dest.at<uchar>(i,j)=0;
                res+="0";
            }
            
        }
    }
    cv::imwrite("average3.png", dest);
    return res;
}

std::string Hash::get_average_hash(cv::Mat &img,int r,int c){
    std::string res;
    res=average_hash(img, r, c);
    return res;
}

float Hash::Ans_average_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=average_hash(img1, r, c);
    str2=average_hash(img2, r, c);
    res=HanmingDistance(str1,str2,r*c);
    return res;
}

//###############block_hash########
std::string Hash::block_hash(cv::Mat &img,int r,int c){
    cv::Mat src=img.clone();
    cv::imwrite("block1.png", src);
    std::string res;
    //    cv::cvtColor(src, src, CV_BGR2GRAY);
    cv::resize(src, src, cv::Size(c*10, r*10), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::imwrite("block2.png", src);
    cv::Mat  mat_mean, mat_stddev;
    cv::meanStdDev(src, mat_mean, mat_stddev);
    double allmean;
    allmean = mat_mean.at<double>(0, 0);
    cv::Mat dest(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            cv::meanStdDev(src(cv::Rect(j*10,i*10,10,10)), mat_mean, mat_stddev);
            float m=mat_mean.at<double>(0,0);
            if(m>allmean){
                res+="1";
                dest.at<uchar>(i,j)=255;
            }
            else {
                res+="0";
                dest.at<uchar>(i,j)=0;
            }
            
        }
    }
    cv::imwrite("block3.png", dest);
    return res;
}

float Hash::Ans_block_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=block_hash(img1, r, c);
    str2=block_hash(img2, r, c);
    res=HanmingDistance(str1,str2,r*c);
    return res;
}

//############difference_hash#####

std::string Hash::difference_hash(cv::Mat &img,int r,int c){
    std::string res;
    cv::Mat src=img.clone();
    cv::imwrite("difference1.png", src);
    //    cv::cvtColor(src, src, CV_BGR2GRAY);
    cv::resize(src, src, cv::Size(c+1, r), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::imwrite("difference2.png", src);
    cv::Mat dest(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(src.at<uchar>(i,j) >src.at<uchar>(i,j+1)){
                res+="1";
                dest.at<uchar>(i,j)=255;
            }
            else{
                res+="0";
                dest.at<uchar>(i,j)=0;
            }
            
        }
    }
    cv::imwrite("difference3.png", dest);
    return res;
}

float Hash::Ans_difference_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=difference_hash(img1, r, c);
    str2=difference_hash(img2, r, c);
    res=HanmingDistance(str1,str2,r*c);
    return res;
}

// ############median_hash####
std::string Hash::median_hash(cv::Mat &img,int r,int c){
    std::string res;
    cv::Mat src=img.clone(),dest=img.clone();
    cv::imwrite("median1.png", src);
    //    cv::cvtColor(src, src, CV_BGR2GRAY);
    //    cv::sortIdx(src, dest, CV_SORT_ASCENDING);
    cv::sort(src, dest, CV_SORT_ASCENDING);
    uint median=dest.at<uchar>(r/2,c-1);
    cv::resize(src, src, cv::Size(c, r), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::Mat dest2(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(src.at<uchar>(i,j)>median){
                res+="1";
                dest2.at<uchar>(i,j)=255;
            }
            else {
                res+="0";
                dest2.at<uchar>(i,j)=0;
            }
        }
    }
    cv::imwrite("median2.png", dest2);
    return res;
}

float Hash::Ans_median_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=median_hash(img1, r, c);
    str2=median_hash(img2, r, c);
    res=HanmingDistance(str1,str2,r*c);
    return res;
    
}

//###########perceptual_hash
std::string Hash::perceptual_hash(cv::Mat &img,int r,int c){
    std::string res;
    cv::imwrite("perceptual1.png", img);
    cv::Mat src=cv::Mat_<double>(img),dest;
    cv::resize(src, src, cv::Size(c*2, r*2), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::imwrite("perceptual2.png", src);
    cv::dct(src, dest);
    cv::imwrite("perceptual3.png", dest);
    //    double mean=0.0;
    //    for(int i=0;i<r;i++){
    //        for(int j=0;j<c;j++){
    //            mean+=dest.at<double>(i,j);
    //        }
    //    }
    //    mean/=r*c;
    dest=dest(cv::Rect(0,0,c,r));
    cv::imwrite("perceptual4.png", dest);
    
    cv::sort(dest, dest, CV_SORT_ASCENDING);
    double median=dest.at<double>(r/2,c-1);
    cv::Mat dest2(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(median<dest.at<double>(i,j)){
                res+="1";
                dest2.at<uchar>(i,j)=255;
            }
            else{
                res+="0";
                dest2.at<uchar>(i,j)=0;
            }
        }
    }
    cv::imwrite("perceptual4.png", dest2);
    return res;
}

float Hash::Ans_perceptual_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=perceptual_hash(img1, r, c);
    str2=perceptual_hash(img2, r, c);
    res=HanmingDistance(str1,str2,r*c);
    return res;
    
}


//#######wavelet_hash
std::string Hash::wavelet_hash(cv::Mat &img,int r,int c){
    std::string res;
    WaveTransform  wmt;
    cv::Mat src=img.clone();
    cv::imwrite("wavelet1.png", src);
    cv::resize(src, src, cv::Size(c*8, r*8), (0, 0), (0, 0), cv::INTER_CUBIC);
    cv::imwrite("wavelet2.png", src);
    cv::Mat dest=wmt.WDT(src, "haar", 3);
    dest=dest(cv::Rect(0,0,c,r));
    //cv::imwrite("/Users/zrain/Desktop/scshot/hash/wavelet1_hash.png", dest);
    cv::sort(dest, dest, CV_SORT_ASCENDING);
    //cv::imwrite("/Users/zrain/Desktop/scshot/hash/wavelet2_hash.png", dest);
    double median=dest.at<double>(r/2,c-1);
    cv::Mat dest2(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(median<dest.at<double>(i,j)){
                dest2.at<uchar>(i,j)=255;
                res+="1";
            }
            else{
                dest2.at<uchar>(i,j)=0;
                res+="0";
            }
        }
    }
        cv::imwrite("waveletABC3.png", dest2);
    return res;
}

float Hash::Ans_wavelet_hash(cv::Mat &img1,cv::Mat &img2,int r,int c){
    float res=0;
    std::string str1,str2;
    str1=wavelet_hash(img1, r, c);
    str2=wavelet_hash(img2, r, c);
    //    std::cout<<str1<<std::endl<<str2<<std::endl;
    res=HanmingDistance(str1,str2,r*c);
    return res;
}










