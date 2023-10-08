use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;


const NEGATIVE_ONE_F64: f64 = -1.0;

const EPSILON2_F64: f64 = 1.0e-8;



const ONE_HALF_F64: f64 = 0.5;





pub fn generate_random_rays_to_center<Z: arrayfire::RealFloating<AggregateOutType = Z, UnaryOutType = Z>  >(
	neuron_pos: &arrayfire::Array<Z>,
	ray_num: u64,
	con_rad: f64,

	start_line: &mut arrayfire::Array<Z>,
	dir_line: &mut arrayfire::Array<Z>
	)
{

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let NEGATIVE_ONE = arrayfire::constant::<f64>(NEGATIVE_ONE_F64,single_dims).cast::<Z>();

	let EPSILON2 = arrayfire::constant::<f64>(EPSILON2_F64,single_dims).cast::<Z>();

	let ONE_HALF = arrayfire::constant::<f64>(ONE_HALF_F64,single_dims).cast::<Z>();

	let con_rad_Z = arrayfire::constant::<f64>(con_rad,single_dims).cast::<Z>();







	let space_dims: u64 = neuron_pos.dims()[1];



	let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);

	*start_line =  arrayfire::tile(neuron_pos, tile_dims);

	*dir_line =  start_line.clone()*NEGATIVE_ONE;




	//Mag of dir_line
	let mut mag2 = arrayfire::pow(dir_line,&TWO,false);
	mag2 = arrayfire::sum(&mag2, 1);




	//Generate random vectors
	let start_line_num =  start_line.dims()[0];
	let rand_dims = arrayfire::Dim4::new(&[start_line_num,space_dims,1,1]);
	let mut rand_vec = (arrayfire::randu::<Z>(rand_dims) - ONE_HALF);
	
	//Normalize random Vector
	let mut mag = arrayfire::pow(&rand_vec,&TWO,false);
	mag = arrayfire::sum(&mag, 1);
	mag = arrayfire::sqrt(&mag) + EPSILON2.clone();

	
	//Scale random vector to connection radius
	rand_vec = arrayfire::div(&rand_vec,&mag,true);
	mag = arrayfire::sqrt(&mag2);
	rand_vec = arrayfire::mul(&rand_vec, &mag, true);
	drop(mag);





	//Vector Projection
	let mut projvec = arrayfire::mul(&rand_vec, dir_line, false);
	projvec = arrayfire::sum(&projvec, 1);

	mag2 = mag2 + EPSILON2.clone();
	projvec = arrayfire::div(&projvec, &mag2, false);
	drop(mag2);

	//Vector rejection
	projvec = rand_vec.clone() -  arrayfire::mul(&projvec, dir_line,true);
	drop(rand_vec);

	//Random scale
	let rand2_dims = arrayfire::Dim4::new(&[start_line_num,1,1,1]);
	let mut rand2_vec = TWO.clone()*arrayfire::randu::<Z>(rand2_dims) ;
	projvec = arrayfire::mul(&projvec, &rand2_vec, true);

	*dir_line = dir_line.clone() + projvec;
	


	//Scale dir line
	let mut mag3 = arrayfire::pow(dir_line ,&TWO,false);
	mag3 = arrayfire::sum(&mag3, 1);
	mag3 = arrayfire::sqrt(&mag3) + EPSILON2.clone();

	*dir_line = con_rad_Z*arrayfire::div(dir_line, &mag3, true);



}











pub fn generate_random_uniform_rays<Z: arrayfire::RealFloating<ProductOutType = Z, UnaryOutType = Z>  >(
	neuron_pos: &arrayfire::Array<Z>,
	ray_num: u64,
	con_rad: f64,

	start_line: &mut arrayfire::Array<Z>,
	dir_line: &mut arrayfire::Array<Z>
	)
{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let TWO_PI = arrayfire::constant::<f64>(TWO_F64*std::f64::consts::PI,single_dims).cast::<Z>();

	let con_rad_Z = arrayfire::constant::<f64>(con_rad,single_dims).cast::<Z>();











	let space_dims: u64 = neuron_pos.dims()[1];






	let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);

	*start_line =  arrayfire::tile(neuron_pos, tile_dims);



	if space_dims == 2
	{
		let start_line_num =  start_line.dims()[0];
		let t_dims = arrayfire::Dim4::new(&[start_line_num,1,1,1]);
		let t = TWO_PI.clone()*arrayfire::randu::<Z>(t_dims);
	
		let x = con_rad_Z.clone()*arrayfire::cos(&t);
		let y = con_rad_Z.clone()*arrayfire::sin(&t);
	
		*dir_line = arrayfire::join(1, &x, &y);
	}
	else
	{
		let start_line_num =  start_line.dims()[0];
		let t_dims = arrayfire::Dim4::new(&[start_line_num,1,1,1]);
		let mut t = TWO_PI.clone()*arrayfire::randu::<Z>(t_dims);
	
		*dir_line = con_rad_Z.clone()*arrayfire::cos(&t);
		for i in 1..(space_dims-1)
		{
			let mut newd = arrayfire::sin(&t);
			
			let newt = TWO_PI.clone()*arrayfire::randu::<Z>(t_dims);
			let lastd = arrayfire::cos(&newt);
			newd = arrayfire::join(1, &newd, &lastd);
			newd = con_rad_Z.clone()*arrayfire::product(&newd,1);


			*dir_line = arrayfire::join(1, dir_line, &newd);
			t = arrayfire::join(1, &t, &newt);
		}

		
		let mut newd = arrayfire::sin(&t);
		newd = con_rad_Z*arrayfire::product(&newd,1);
		*dir_line = arrayfire::join(1, dir_line, &newd);
	
	}
	

}







