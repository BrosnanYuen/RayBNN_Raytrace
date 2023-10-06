use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;







pub fn tileDown<Z: arrayfire::HasAfEnum >(
	repeat_num: u64,

	input_arr: &mut arrayfire::Array<Z>
	)
{
	let space_dims: u64 = input_arr.dims()[1];

	let input_arr_num: u64 = input_arr.dims()[0];

	let tile_dims = arrayfire::Dim4::new(&[1,repeat_num,1,1]);

	*input_arr = arrayfire::tile(input_arr, tile_dims);

	*input_arr = arrayfire::transpose(input_arr, false);

	let dims = arrayfire::Dim4::new(&[space_dims, repeat_num*input_arr_num , 1 , 1]);
	*input_arr = arrayfire::moddims(input_arr, dims);

	*input_arr = arrayfire::transpose(input_arr, false);


}





