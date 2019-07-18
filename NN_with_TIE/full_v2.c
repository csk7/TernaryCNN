

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TIE



#include"max_pooling.h"
#include"activation_fun.h"
#include"fully_connected.h"
#include"convolution.h"


 #ifdef TIE
    #include "conv_tie_v1.h"
 #endif


int* unrolling(int*** image,int ch,int r,int c)
{
	int l=0;
	int* out_vec;
	int i,j,k;
	out_vec = (int*) malloc(ch*r*c*sizeof(int));
	for(i = 0;i<ch;i=i+1)
	{
		for(j=0;j<r;j=j+1)
			{
				for(k=0;k<c;k=k+1)
				{
                     out_vec[l] = image[i][j][k];
                     l++;
				}
			}
	}
	return out_vec;
}



int max_10(int* vec,int length)
{
	length =10;
   int max_out = vec[0];
   int pos = 1;
   int i;
   for(i=1;i<length;i=i+1)
   {
        if (vec[i] > max_out)
        {
        	pos = i;
        	max_out = vec[i];
        }
   }
   return pos;
}


int main()
{
	
	int r=28;
	int c=28;
	int filter_size = 5;
	int*** image_x1;
	int*** image_x2;
	int*** image_x3;
	int*** image_x4;
	int*** image_x5;
	int*** image_x6;
    int* unrolled_vector;
	int* un_vect1;
	int* un_vect2;
	int* un_vect3;



	int*** image;
 	int**** weight_conv1;
	int* bias_conv1;
	int**** weight_conv2;
	int* bias_conv2;
	int** weight_fc1;
	int* bias_fc1;
	int** weight_fc2;
	int* bias_fc2;
	
	#include"read_v1.h"
	#include"read_v2.h"
	#include"read_v3.h"
	#include"read_v4.h"
	#include"images1.h"

	int final_out;
    int i,j,k,k2;
    
    image = (int***) malloc(1*sizeof(int**));
    
	for(i=0;i<1;i=i+1)
	{
		image[i] = (int**) malloc(28*sizeof(int*));
		for(j=0;j<28;j=j+1)
		{
			image[i][j] = (int*) malloc(28*sizeof(int));
			
			for(k=0;k<28;k=k+1)
			{
				image[i][j][k] = 	image1[i][j][k];
				//printf ("Decimals: %d %ld\n", 1977, image[i][j][k]);
				
			}
			 //printf ("Decimals: %d %ld\n", j, 650000);
	    }
	    
   
   }
    
    //printf ("Decimals: %d %ld %ld\n", i, j, k);
	
   	weight_conv1 = (int**** ) malloc(32*sizeof(int***));
	for(i=0;i<32;i=i+1)
	{
		weight_conv1[i] = (int***) malloc(1*sizeof(int**));
		for(j=0;j<1;j=j+1)
		{
			weight_conv1[i][j] = (int**) malloc(filter_size*sizeof(int*));
			for(k=0;k<filter_size;k=k+1)
			{
				weight_conv1[i][j][k] = (int*) malloc(filter_size*sizeof(int));
				for(k2=0;k2<filter_size;k2=k2+1)
				{
					weight_conv1[i][j][k][k2] = conv1_weight[i][j][k][k2];
					
				}
	        }
		}
	}
    
	
	bias_conv1 = (int*) malloc(32*sizeof(int));
	for(i=0;i<32;i=i+1)
		bias_conv1[i] = conv1_bias[i];
		
    
   	weight_conv2 = (int****) malloc(64*sizeof(int***));
	for(i=0;i<64;i=i+1)
	{
		weight_conv2[i] = (int***) malloc(32*sizeof(int**));
		
		for(j=0;j<32;j=j+1)
		{
			weight_conv2[i][j] = (int**) malloc(filter_size*sizeof(int*));
			for(k=0;k<filter_size;k=k+1)
			{
				weight_conv2[i][j][k] = (int*) malloc(filter_size*sizeof(int));
				for(k2=0;k2<filter_size;k2=k2+1)
				{
					weight_conv2[i][j][k][k2] = conv2_weight[i][j][k][k2];
				    
				}
				
	        }
		}
	}
    	
	bias_conv2 = (int*) malloc(64*sizeof(int));
	for(i=0;i<64;i=i+1)
		bias_conv2[i] = conv2_bias[i];
   
    weight_fc1 = (int**) malloc(50*sizeof(int*));
	for(i=0;i<50;i=i+1)
	{
		weight_fc1[i] = (int*) malloc(1024*sizeof(int));
		for(j=0;j<1024;j=j+1)
		{
			weight_fc1[i][j] = fc1_weight[i][j];
		}
   }
     
   	bias_fc1 = (int*) malloc(50*sizeof(int));
	for(i=0;i<50;i=i+1)
	{
		bias_fc1[i] = fc1_bias[i];
    }
    
    weight_fc2 = (int**) malloc(10*sizeof(int*));
	for(i=0;i<10;i=i+1)
	{
		weight_fc2[i] = (int*) malloc(50*sizeof(int));
		for(j=0;j<50;j=j+1)
		{
			weight_fc2[i][j] = fc2_weight[i][j];
			
			
		}
   }
   
    
   	bias_fc2 = (int*) malloc(10*sizeof(int));
   	
	for(i=0;i<10;i=i+1)
	{
		bias_fc2[i] = fc2_bias[i];
		
    }
    
   
    
	xt_profile_enable();   //START

    image_x1 = conv2d(image,weight_conv1,bias_conv1,1,32,&r,&c,5);
    
    
	for(i=0;i<32;i=i+1)
	{
		
		for(j=0;j<1;j=j+1)
		{
			
			for(k=0;k<filter_size;k=k+1)
			{
					free(weight_conv1[i][j][k]);				
	        }
	        free(weight_conv1[i][j]);
		}
		free(weight_conv1[i]);
	}
	free(weight_conv1);
	
	
    free(bias_conv1);


    image_x2  = max_pool(image_x1,32,&r,&c,2);
    
    

    image_x3 = activation(image_x2,32,r,c);
    
    

    image_x4 = conv2d(image_x3,weight_conv2,bias_conv2,32,64,&r,&c,5);

    
	for(i=0;i<64;i=i+1)
	{
		
		
		for(j=0;j<32;j=j+1)
		{
			
			for(k=0;k<filter_size;k=k+1)
			{	
            	free(weight_conv2[i][j][k]);	
	        }
	        free(weight_conv2[i][j]);
		}
		free(weight_conv2[i]);
	}
	free(weight_conv2);
    	
	free(bias_conv2);

    image_x5  = max_pool(image_x4,64,&r,&c,2);

    image_x6 = activation(image_x5,64,r,c);
    
    unrolled_vector = unrolling(image_x6,64,r,c);


    un_vect1 = fully_connect(unrolled_vector,weight_fc1,bias_fc1,50,1024);

    
	for(i=0;i<50;i=i+1)
	{
		free(weight_fc1[i]);
    }
   free(weight_fc1);
     
   	free(bias_fc1);


    un_vect2 = activation_1d(un_vect1,50);

    un_vect3 = fully_connect(un_vect2,weight_fc2,bias_fc2,10,50);
    

	for(i=0;i<10;i=i+1)
	{
			free(weight_fc2[i]);
    }
    free(weight_fc2);
    
   	free(bias_fc2);
   	

    final_out = max_10(un_vect3,10);

    printf ("Decimals: %d\n", final_out);

    xt_profile_disable(); //END
    
    
	for(i=0;i<1;i=i+1)
	{	
		for(j=0;j<28;j=j+1)
		{
			free(image[i][j]);	
	    }
	    free(image[i]);
   
   }
   free(image);

}
