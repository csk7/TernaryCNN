int*** slicing(int*** image,int r,int c,int channels,int pos_x,int pos_y,int filter_size)
{
	int*** sub_image;
	int i,j,k,ch;
	sub_image = (int***) malloc(channels*sizeof(int**));
	for(i=0;i<channels;i= i+1)
	{	
		sub_image[i] = (int**) malloc(filter_size*sizeof(int*));
		for(k=0;k<filter_size;k=k+1)
			sub_image[i][k] = (int*) malloc(filter_size*sizeof(int));
    }
    
    for(i=pos_x;i<(pos_x + filter_size);i=i+1)
		for(j=pos_y;j<(pos_y + filter_size);j=j+1)
			for(ch=0;ch<channels;ch=ch+1)
				sub_image[ch][i - pos_x][j - pos_y] = image[ch][i][j];
	
	return sub_image;			
}

int elem_mult2d(int** a,int** b,int r1,int c1)
{
	int sum = 0;
	int i,j;
	for(i=0;i<r1;i=i+1)
		for(j=0;j<c1;j=j+1)
			sum += a[i][j] * b[i][j];
	
	return sum;		
}

int out_single_channel(int*** a,int*** b,int channels,int r1,int c1,int bias)
{

	int out = bias;
    int i;
	
	for(i=0;i<channels;i=i+1)
	{
		out += elem_mult2d(a[i],b[i],r1,c1);	
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
				
				sub_image=slicing(image,*r,*c,in_ch,i,j,filter_size);
					
				out[ch_count][i][j] = out_single_channel(sub_image,weight[ch_count],in_ch,filter_size,filter_size,bias[ch_count]);
				
					
			}
		}
			
	}
	
	*r = out_r;
	*c = out_c;
	//cout<<"Hello";
	return out;
}













