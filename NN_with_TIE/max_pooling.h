#define TIE
 #ifdef TIE
    #include "conv_tie_v1.h"
 #endif

int max4(int*** image, int channel,int j_begin,int k_begin,int stride)
{
	stride = 2;
	int j,k;
	int max_val=image[channel][j_begin][k_begin];
   	for(j=j_begin;j<(j_begin + 2);j=j+1)
   	{
   		for(k=k_begin;k<(k_begin + 2);k=k+1)
   		{
   			if (image[channel][j][k]>max_val)
   			{
   			  max_val = image[channel][j][k];	
			}
		}
	}
	return max_val;
}


int*** max_pool(int*** image, int channels,int *r,int *c,int stride)
{
	stride = 2;
	int i,j,k;
	int*** out;
	int new_r = (*r)/stride;
	int new_c = (*c)/stride;
	
	out = (int***) malloc(channels*sizeof(int**));
	for(i=0;i<channels;i=i+1)
	{
		out[i] = (int**) malloc(new_r*sizeof(int*));
		for(j=0;j<new_r;j=j+1)
		{
			out[i][j] = (int*) malloc(new_c*sizeof(int));
			for(k = 0;k<new_c;k=k+1)
				out[i][j][k] = 0;
	    }   
	}
	
	for (i=0;i<channels;i=i+1)
	{
	
		for (j = 0; j<((*r)-2);j=j+4)
		{
			for(k =0;k<((*c)-2);k=k+4)
			{
				//out[i][j/2][k/2] = max4(image,i,j,k,stride);
				WUR_reg_O(image[i][j][k]);
				WUR_reg_P(image[i][j][k+1]);
				WUR_reg_Q(image[i][j+1][k]);
				WUR_reg_R(image[i][j+1][k+1]);
				out[i][j/2][k/2] = maxpool_4_all();

				WUR_reg_O1(image[i][j+2][k+2]);
				WUR_reg_P1(image[i][j+2][k+3]);
				WUR_reg_Q1(image[i][j+3][k+2]);
				WUR_reg_R1(image[i][j+3][k+3]);
				out[i][j/2][k/2] = maxpool_4_all_1();
			}
		
		}
    }
    *r = new_r;
    *c = new_c;
    return out;
}

