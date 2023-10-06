use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;





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







pub fn filter_rays<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	con_rad: Z,

	target_input_pos: &arrayfire::Array<Z>,

	input_pos: &mut arrayfire::Array<Z>,
	input_idx: &mut arrayfire::Array<i32>,
	)
{

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();




	let input_diff = arrayfire::sub(target_input_pos, input_pos, true);



	let con_rad_sq = con_rad*con_rad;

	let mut mag2 = arrayfire::pow(&input_diff,&TWO,false);
	mag2 = arrayfire::sum(&mag2, 1);

	//  (con_rad_sq >= mag2 )
	let CMPRET = arrayfire::ge(&con_rad_sq, &mag2, false);
	drop(mag2);

	//Lookup  1 >= dir_line  >= 0
	let idx_intersect = arrayfire::locate(&CMPRET);
	drop(CMPRET);

	*input_pos = arrayfire::lookup(input_pos, &idx_intersect, 0);

	*input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);


}

















