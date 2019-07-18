#define TIE
 #ifdef TIE
    #include "conv_tie_v1.h"
 #endif

int*** slicing(int*** image,int r,int c,int channels,int pos_x,int pos_y,int filter_size)
{
		int*** sub_image;
		int i1,j1,k1,ch1;
		sub_image = (int***) malloc(channels*sizeof(int**));
		for(i1=0;i1<channels;i1= i1+1)
		{
			sub_image[i1] = (int**) malloc(filter_size*sizeof(int*));
			for(k1=0;k1<filter_size;k1=k1+1)
				sub_image[i1][k1] = (int*) malloc(filter_size*sizeof(int));
		}

		for(i1=pos_x;i1<(pos_x + filter_size);i1=i1+1)
			for(j1=pos_y;j1<(pos_y + filter_size);j1=j1+1)
				for(ch1=0;ch1<channels;ch1=ch1+1)
					sub_image[ch1][i1 - pos_x][j1 - pos_y] = image[ch1][i1][j1];
	
	return sub_image;			
}

int elem_mult2d(int** a,int n_r,int n_c,int** b,int r1,int c1)
{
	int sum;
    int n_c1 = n_c+1;
    int n_c2 = n_c+2;
    int n_c3 = n_c+3;
    int n_c4 = n_c+4;

    int n_r1 = n_r+1;
    int n_r2 = n_r+2;
    int n_r3 = n_r+3;
    int n_r4 = n_r+4;

	WUR_reg_A(a[n_r][n_c]);
	WUR_reg_B(a[n_r][n_c1]);
	WUR_reg_C(a[n_r][n_c2]);
	WUR_reg_D(a[n_r][n_c3]);
	WUR_reg_E(a[n_r][n_c4]);

	WUR_reg_J(b[0][0]);
	WUR_reg_K(b[0][1]);
	WUR_reg_L(b[0][2]);
	WUR_reg_M(b[0][3]);
	WUR_reg_N(b[0][4]);

	WUR_reg_A1(a[n_r+1][n_c]);
	WUR_reg_B1(a[n_r1][n_c1]);
	WUR_reg_C1(a[n_r1][n_c2]);
	WUR_reg_D1(a[n_r1][n_c3]);
	WUR_reg_E1(a[n_r1][n_c4]);

	WUR_reg_J1(b[1][0]);
	WUR_reg_K1(b[1][1]);
	WUR_reg_L1(b[1][2]);
	WUR_reg_M1(b[1][3]);
	WUR_reg_N1(b[1][4]);

	WUR_reg_A2(a[n_r2][n_c]);
	WUR_reg_B2(a[n_r2][n_c1]);
	WUR_reg_C2(a[n_r2][n_c2]);
	WUR_reg_D2(a[n_r2][n_c3]);
	WUR_reg_E2(a[n_r2][n_c4]);

	WUR_reg_J2(b[2][0]);
	WUR_reg_K2(b[2][1]);
	WUR_reg_L2(b[2][2]);
	WUR_reg_M2(b[2][3]);
	WUR_reg_N2(b[2][4]);

	WUR_reg_A3(a[n_r3][n_c]);
	WUR_reg_B3(a[n_r3][n_c1]);
	WUR_reg_C3(a[n_r3][n_c2]);
	WUR_reg_D3(a[n_r3][n_c3]);
	WUR_reg_E3(a[n_r3][n_c4]);

	WUR_reg_J3(b[3][0]);
	WUR_reg_K3(b[3][1]);
	WUR_reg_L3(b[3][2]);
	WUR_reg_M3(b[3][3]);
	WUR_reg_N3(b[3][4]);

	WUR_reg_A4(a[n_r4][n_c]);
	WUR_reg_B4(a[n_r4][n_c1]);
	WUR_reg_C4(a[n_r4][n_c2]);
	WUR_reg_D4(a[n_r4][n_c3]);
	WUR_reg_E4(a[n_r4][n_c4]);

	WUR_reg_J4(b[4][0]);
	WUR_reg_K4(b[4][1]);
	WUR_reg_L4(b[4][2]);
	WUR_reg_M4(b[4][3]);
	WUR_reg_N4(b[4][4]);

    sum = tie_rowCalculation1() + tie_rowCalculation2()+ tie_rowCalculation3()+ tie_rowCalculation4()+ tie_rowCalculation5();
	
	return sum;		
}

int out_single_channel(int*** a,int n_r,int n_c,int*** b,int channels,int r1,int c1,int bias)
{

	int out = bias;
    int i;
	
	for(i=0;i<channels;i=i+1)
	{
		out += elem_mult2d(a[i],n_r,n_c,b[i],r1,c1);
	}


	return out;	
}

int*** conv2d(int*** image,int**** weight,int* bias,int in_ch, int out_ch,int *r,int *c,int filter_size)
{
	int*** out;
	int*** sub_image;
	int out_r = (*r)-filter_size + 1;
	int out_c = (*c)-filter_size + 1;
	int i,j,k;
	out = (int***) malloc (out_ch*sizeof(int**));
	for(i=0;i<out_ch;i=i+1)
	{
		out[i] = (int**) malloc(out_r*sizeof(int*));
		for(j=0;j<out_r;j=j+1)
		{
			out[i][j] = (int*) malloc(out_c*sizeof(int));
			for(k = 0;k<out_c;k=k+1)
				out[i][j][k] = 0;
	    }   
	}
	
	int ch_count;
	for(ch_count=0; ch_count<out_ch;ch_count=ch_count+1)
	{
	
		for(i=0;i<=((*r)-filter_size);i=i+1)
		{
			for(j=0;j<=((*c)-filter_size);j=j+1)
			{
				
				//sub_image=slicing(image,*r,*c,in_ch,i,j,filter_size);
					
				out[ch_count][i][j] = out_single_channel(image,i,j,weight[ch_count],in_ch,filter_size,filter_size,bias[ch_count]);
				
					
			}
		}
			
	}
	
	*r = out_r;
	*c = out_c;
	//cout<<"Hello";
	return out;
}













