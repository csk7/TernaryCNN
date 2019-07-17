
int* fully_connect(int* image,int** weight,int* bias, int r,int c)
{
	int i,j;
	int* out_image;
	out_image = (int*) malloc(r*sizeof(int));
	for(i=0;i<r;i= i+1)
	{
		out_image[i] = bias[i];
		for(j=0;j<c;j=j+1)
		{
			out_image[i] += image[j]*weight[i][j];
		}
	}
	return out_image;
}


