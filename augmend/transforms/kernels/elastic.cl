
#ifndef ADDRESS_MODE
#define ADDRESS_MODE CLK_ADDRESS_CLAMP
#endif

#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif



__kernel void zoom_and_transform2(__read_only image2d_t img,
								  __read_only image2d_t dy,
								  __read_only image2d_t dx,
								  __global TYPENAME* output)
{
  const sampler_t sampler_dx = CLK_NORMALIZED_COORDS_TRUE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_LINEAR;

  const sampler_t sampler_img = CLK_NORMALIZED_COORDS_FALSE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	SAMPLERFILTER;

  
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  uint x = get_global_id(0);
  uint y = get_global_id(1);

  float xnorm = 1.f*x/Nx;
  float ynorm = 1.f*y/Ny;

  float _dx = read_imagef(dx,sampler_dx,
						  (float2)(xnorm,ynorm)).x;
  float _dy = read_imagef(dy,sampler_dx,
						  (float2)(xnorm,ynorm)).x;


  TYPENAME pix = READ_IMAGE(img,sampler_img,
							(float2)(x+_dx,
									 y+_dy)).x;


  output[x+Nx*y] = pix;

}


__kernel void zoom_and_transform3(__read_only image3d_t img,
								  __read_only image3d_t dz,
								  __read_only image3d_t dy,
								  __read_only image3d_t dx,
								  __global TYPENAME* output)
{
  const sampler_t sampler_dx = CLK_NORMALIZED_COORDS_TRUE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_LINEAR;

  const sampler_t sampler_img = CLK_NORMALIZED_COORDS_FALSE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	SAMPLERFILTER;

  
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint z = get_global_id(2);

  float xnorm = 1.f*x/Nx;
  float ynorm = 1.f*y/Ny;
  float znorm = 1.f*z/Nz;

  float _dx = read_imagef(dx,sampler_dx,
						  (float4)(xnorm,ynorm,znorm,0.f)).x;
  float _dy = read_imagef(dy,sampler_dx,
						  (float4)(xnorm,ynorm,znorm,0.f)).x;
  float _dz = read_imagef(dz,sampler_dx,
						  (float4)(xnorm,ynorm,znorm,0.f)).x;


  TYPENAME pix = READ_IMAGE(img,sampler_img,
							(float4)(x+_dx,
									 y+_dy,
									 z+_dz,
									 0.f)).x;


  output[x+Nx*y+Nx*Ny*z] = pix;

}
