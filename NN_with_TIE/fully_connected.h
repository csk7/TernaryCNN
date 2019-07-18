#define TIE
 #ifdef TIE
    #include "conv_tie_v1.h"
 #endif

int* fully_connect(int* image,int** weight,int* bias, int r,int c)
{
	int i,j;
	int* out_image;
	out_image = (int*) malloc(r*sizeof(int));
	for(i=0;i<r;i= i+1)
	{
		out_image[i] = bias[i];
		for(j=0;j<(c-3);j=j+4)
		{
			WUR_reg_A(image[j]);
			WUR_reg_B(image[j+1]);
			WUR_reg_C(image[j+2]);
			WUR_reg_D(image[j+3]);


			WUR_reg_J(weight[i][j]);
			WUR_reg_K(weight[i][j+1]);
			WUR_reg_L(weight[i][j+2]);
			WUR_reg_M(weight[i][j+3]);

			out_image[i] += fc4_t2();

			/*if((j+2)>=c)
			{
				WUR_reg_S(image[j]);
				WUR_reg_T(image[j+1]);
				WUR_reg_U(weight[i][j]);
				WUR_reg_W(weight[i][j+1]);
				out_image[i] += fc_t1();
			}*/
		}
	}
	return out_image;
}


