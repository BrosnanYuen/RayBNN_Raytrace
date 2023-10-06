#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_dataset_csv() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);




    let (mut read_test,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f64>(
    	"./test_data/read_test.dat"
    );


	let mut read_act: Vec<f64> = vec![
		-0.004866,-0.0018368,0.0049874,0.0023202,-4.9179e-05,-0.0033278,
		-0.0082358,-0.006966,-0.0033703,0.0038264,0.0047417,0.0017643,
		0.0013178,-0.00061582,0.008669,3.5362e-05,-0.00080587,0.0044014,
		0.00012772,-0.00088359,-0.0072174,0.0043621,0.0046395,2.6826e-05
	];

	read_act = read_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	read_test = read_test.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();


	assert_eq!(read_test, read_act);
    assert_eq!(metadata["dim0"], 4);
    assert_eq!(metadata["dim1"], 6);







    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i64>(
    	"./test_data/read_test2.dat"
    );

    let mut read_act2: Vec<i64> = vec![
        233,-4233,234,631,
        24, 222,-1,23,
        45,3,1,100,
        -2,3,  5,61,
        344,222,33,-10,
        751,-32,12,92,
        431,585,-4,215
    ];
    assert_eq!(read_test2, read_act2);
    assert_eq!(metadata["dim0"], 7);
    assert_eq!(metadata["dim1"], 4);




    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i32>(
    	"./test_data/read_test2.dat"
    );

    let mut read_act2: Vec<i32> = vec![
        233,-4233,234,631,
        24, 222,-1,23,
        45,3,1,100,
        -2,3,  5,61,
        344,222,33,-10,
        751,-32,12,92,
        431,585,-4,215
    ];
    assert_eq!(read_test2, read_act2);
    assert_eq!(metadata["dim0"], 7);
    assert_eq!(metadata["dim1"], 4);








    /* 
    let arr = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/read_test.dat"
    );

    arrayfire::print_gen("arr".to_string(), &arr,Some(6));
    */



    let read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/read_test.dat"
    );

    assert_eq!(read_test.dims()[0], 4);
    assert_eq!(read_test.dims()[1], 6);


	let row0 = arrayfire::row(&read_test,0);

	let mut row0_cpu = vec!(f64::default();row0.elements());
	row0.host(&mut row0_cpu);

	let mut row0_act: Vec<f64> = vec![-0.004866,-0.0018368,0.0049874,0.0023202,-4.9179e-05,-0.0033278];

	row0_act = row0_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row0_cpu = row0_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row0_cpu, row0_act);










	let row1 = arrayfire::row(&read_test,1);

	let mut row1_cpu = vec!(f64::default();row1.elements());
	row1.host(&mut row1_cpu);

	let mut row1_act: Vec<f64> = vec![-0.0082358,-0.006966,-0.0033703,0.0038264,0.0047417,0.0017643];

	row1_act = row1_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row1_cpu = row1_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row1_cpu, row1_act);











	let row2 = arrayfire::row(&read_test,2);

	let mut row2_cpu = vec!(f64::default();row2.elements());
	row2.host(&mut row2_cpu);

	let mut row2_act: Vec<f64> = vec![0.0013178,-0.00061582,0.008669,3.5362e-05,-0.00080587,0.0044014];

	row2_act = row2_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row2_cpu = row2_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row2_cpu, row2_act);







	let row3 = arrayfire::row(&read_test,3);

	let mut row3_cpu = vec!(f64::default();row3.elements());
	row3.host(&mut row3_cpu);

	let mut row3_act: Vec<f64> = vec![0.00012772,-0.00088359,-0.0072174,0.0043621,0.0046395,2.6826e-05];

	row3_act = row3_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row3_cpu = row3_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row3_cpu, row3_act);








    let read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i64>(
    	"./test_data/read_test2.dat"
    );

    assert_eq!(read_test.dims()[0], 7);
    assert_eq!(read_test.dims()[1], 4);


	let row0 = arrayfire::row(&read_test,0);

	let mut row0_cpu = vec!(i64::default();row0.elements());
	row0.host(&mut row0_cpu);

	let mut row0_act: Vec<i64> = vec![233,-4233,234,631,];

	assert_eq!(row0_cpu, row0_act);







	let row1 = arrayfire::row(&read_test,1);

	let mut row1_cpu = vec!(i64::default();row1.elements());
	row1.host(&mut row1_cpu);

	let mut row1_act: Vec<i64> = vec![24, 222,-1,23,];


	assert_eq!(row1_cpu, row1_act);













	let row2 = arrayfire::row(&read_test,6);

	let mut row2_cpu = vec!(i64::default();row2.elements());
	row2.host(&mut row2_cpu);

	let mut row2_act: Vec<i64> = vec![431,585,-4,215];

	assert_eq!(row2_cpu, row2_act);


}
