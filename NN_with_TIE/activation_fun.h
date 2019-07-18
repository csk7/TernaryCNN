
#define TIE
 #ifdef TIE
    #include "conv_tie_v1.h"
 #endif
int function_sign(int num,int threshold)
{
	
	if(num >= threshold)
	   return 1;
	else if (num <= (-1*threshold)) 
	   return -1;
	else
	   return 0;
	   	        
}

int*** activation(int*** image,int ch, int n_r,int n_c)
{
	int*** out;
	int i,j,k;

	out = (int***) malloc(ch*sizeof(int**));
	for(i=0;i<ch;i=i+1)
	{
		out[i] = (int**) malloc(n_r*sizeof(int*));
		for(j=0;j<n_r;j=j+1)
		{
			out[i][j] = (int*) malloc(n_c*sizeof(int));
			for(k = 0;k<n_c;k=k+1)
				out[i][j][k] = 0;
	    }   
	}
	
	for(i = 0;i<ch;i=i+1)
	{
		for(j=0;j<n_r;j=j+1)
		{
			for(k=0;k<(n_c-1);k=k+2)
			{
				//out[i][j][k] = function_sign(image[i][j][k],1);
				//WUR_reg_H(image[i][j][k]);
				//WUR_reg_thresh_pos(1);
				//WUR_reg_thresh_neg(-1);
				out[i][j][k] = thresholding_tie(image[i][j][k]);
				out[i][j][k+1] = thresholding_tie(image[i][j][k+1]);
			}
		}
	}
	return out;
}

int* activation_1d(int* vec,int length)
{
	//WUR_reg_thresh_pos(1);
	//WUR_reg_thresh_neg(-1);
	int* out_vec;
	int i;
	out_vec = (int*) malloc(length*sizeof(int));
	for(i=0;i<(length-1);i=i+2)
	{
		//out_vec[i] =  function_sign(vec[i],1);
		//WUR_reg_H(vec[i]);
		out_vec[i] = thresholding_tie(vec[i]);
		out_vec[i+1] = thresholding_tie(vec[i+1]);
	}
	return out_vec;
}

