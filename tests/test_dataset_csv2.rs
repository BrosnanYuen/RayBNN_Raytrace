#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use rand::{distributions::Standard, Rng};
use std::collections::HashMap;

#[test]
fn test_dataset_csv2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);





    let mut metadata: HashMap<String,u64> = HashMap::new();
	let randvec: Vec<i32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0".to_string(), 11);
    metadata.insert("dim1".to_string(), 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<i32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata["dim0"], 11);
    assert_eq!(metadata["dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");











    let mut metadata: HashMap<String,u64> = HashMap::new();
	let randvec: Vec<u32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0".to_string(), 11);
    metadata.insert("dim1".to_string(), 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<u32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<u32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata["dim0"], 11);
    assert_eq!(metadata["dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");














    let mut metadata: HashMap<String,u64> = HashMap::new();
	let randvec: Vec<f32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0".to_string(), 11);
    metadata.insert("dim1".to_string(), 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<f32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata["dim0"], 11);
    assert_eq!(metadata["dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");










    let mut metadata: HashMap<String,u64> = HashMap::new();
    let write_vec: Vec<i32> = vec![
        1,-2,3,-4,
        -5,6,-7,8,
        9,10,11,12,
        13,14,15,16,
        -17,18,-19,20,
        21,-22,23,-24,
        25,26,27,28
    ];

    metadata.insert("dim0".to_string(), 7);
    metadata.insert("dim1".to_string(), 4);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<i32>(
		"./test_write.csv",
		&write_vec,
        &metadata
	);


    let mut read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i32>(
    	"./test_write.csv"
    );

    assert_eq!(read_test.dims()[0], 7);
    assert_eq!(read_test.dims()[1], 4);

    std::fs::remove_file("./test_write.csv");

    read_test = arrayfire::sum(&read_test, 0);

    //arrayfire::print_gen("read_test".to_string(), &read_test,Some(6));

    let mut row0_cpu = vec!(i32::default();read_test.elements());
	read_test.host(&mut row0_cpu);

    let row0_act = vec![47,  50,  53,  56];
    assert_eq!(row0_cpu, row0_act);












	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randn::<f64>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<f64>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");









	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randu::<u32>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<u32>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<u32>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");














	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randu::<i32>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<i32>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i32>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");















	let (hashdata,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_hash_cpu::<f64>(
    	"./test_data/dataloader.csv",
    	7,
		5
    );


	let mut vec0_act: Vec<f64> = vec![-12.6505601415073,-13.7020069785047,5.3976325705858,-3.91159125382991,6.99930942955802,18.7630014654607,6.0862313994853,
	2.15460520182276,7.71634326416303,-3.90361103111226,7.49377623166851,-6.2996155894013,-1.80937151778845,0.661399578708471,
	19.0394635060288,-9.12847199674459,-3.51553641750281,4.00633350315077,1.23169734070768,6.27512120829686,-7.56972745960131,
	3.98347774640108,3.66190048424611,14.9210187053127,11.2606999846992,-18.3209459916153,9.64734120627594,0.158590453738361,
	3.41373613489615,9.52355888661374,-13.1183038867781,8.36157079925919,14.4536171051847,3.8518859272116,2.13309717540695];

	let mut vec0_pred = hashdata[&0].clone();
	vec0_pred = vec0_pred.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	vec0_act = vec0_act.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	assert_eq!(vec0_pred, vec0_act);









	let mut vec1_act: Vec<f64> = vec![-4.34169112696739,6.27059967177088,2.75538063364082,11.9788842598508,-3.29090760994155,11.8095278538105,-11.6939507448636,
	2.6918626998768,13.8263350423676,-16.488471768989,1.68062132075419,-9.03849974002218,6.23213137847597,4.92182244870785,
	5.32347012624477,13.0607711081905,13.2049801445265,1.52798543848733,0.460553165771882,-4.84146183664087,-5.22293276290175,
	7.26387557408945,-3.50907736597753,11.6171630239342,13.0777340530544,-11.286468646321,-15.0537335138609,2.05759412040277,
	-0.728087417235042,5.57056157433969,3.01162083483514,-18.2365546241575,2.791423822984,21.5859294855571,6.3621334095424];


	let mut vec1_pred = hashdata[&1].clone();
	vec1_pred = vec1_pred.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	vec1_act = vec1_act.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	assert_eq!(vec1_pred, vec1_act);













	let mut vec2_act: Vec<f64> = vec![0.635454162428137,-6.44377806404219,2.2916278683952,-2.0458640551401,-2.60664996528185,-3.04530088774459,-6.04654059246837,
	23.2369088660305,-10.2178852139256,-21.8628392070328,-9.7059444980866,24.7777870918415,4.03359722310648,0.117436387943036,
	-19.5835397049619,12.2482593349371,8.81172322675923,-0.825848027076484,14.384653873296,8.63166955145978,6.50062757534178,
	25.294456804569,3.00049849075251,-18.924349199659,9.48679262048621,-21.968865031845,4.13596996734002,-15.358591425043,
	6.37563700613106,2.45081085390593,-20.192803066433,11.2572714924176,7.41326931727609,-6.04726212798032,13.3597590824548];

	let mut vec2_pred = hashdata[&2].clone();
	vec2_pred = vec2_pred.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	vec2_act = vec2_act.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	assert_eq!(vec2_pred, vec2_act);











	let mut vec3_act: Vec<f64> = vec![13.0006081958912,12.1112726565386,3.7993371029844,-7.14943642680137,-10.9813787581107,-1.67714582451856,27.2446114493936,
	-13.6416447498706,8.94583697409052,-6.48480983570505,-1.71069912683167,0.449448201191942,1.75110052776893,12.5940050940798,
	-2.90034564205221,6.50201173874535,7.11490621423374,4.3702487413005,4.64404073318729,17.7895849150403,-0.157719950094085,
	0.219358362346906,15.50251228305,9.76977805684029,0.0813662424642665,10.4773402880812,-5.60815690006961,7.28622679859396,
	7.22715691220478,12.2978945855772,11.5773254520536,6.01818555261006,0.269530505970931,10.8466846728782,-7.35984778513393];

	let mut vec3_pred = hashdata[&3].clone();
	vec3_pred = vec3_pred.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	vec3_act = vec3_act.par_iter().map(|x|  (x * 1.0e5).round() / 1.0e5 ).collect::<Vec<f64>>();

	assert_eq!(vec3_pred, vec3_act);












	let arr_dims = arrayfire::Dim4::new(&[5, 7, 1, 1]);
	let (hashdata,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_hash_arrayfire::<f64>(
    	"./test_data/dataloader.csv",
		arr_dims,
    );

	assert_eq!(hashdata[&0].dims()[0], 5);
    assert_eq!(hashdata[&0].dims()[1], 7);

	let read_test = hashdata[&0].clone();

	let row0 = arrayfire::row(&read_test,0);

	let mut row0_cpu = vec!(f64::default();row0.elements());
	row0.host(&mut row0_cpu);

	let mut row0_act: Vec<f64> = vec![-12.6505601415073,-13.7020069785047,5.3976325705858,-3.91159125382991,6.99930942955802,18.7630014654607,6.0862313994853];

	row0_act = row0_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row0_cpu = row0_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row0_cpu, row0_act);











	let row1 = arrayfire::row(&read_test,1);

	let mut row1_cpu = vec!(f64::default();row1.elements());
	row1.host(&mut row1_cpu);

	let mut row1_act: Vec<f64> = vec![2.15460520182276,7.71634326416303,-3.90361103111226,7.49377623166851,-6.2996155894013,-1.80937151778845,0.661399578708471];

	row1_act = row1_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row1_cpu = row1_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row1_cpu, row1_act);










	let row2 = arrayfire::row(&read_test,4);

	let mut row2_cpu = vec!(f64::default();row2.elements());
	row2.host(&mut row2_cpu);

	let mut row2_act: Vec<f64> = vec![3.41373613489615,9.52355888661374,-13.1183038867781,8.36157079925919,14.4536171051847,3.8518859272116,2.13309717540695];

	row2_act = row2_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	row2_cpu = row2_cpu.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	assert_eq!(row2_cpu, row2_act);



	//arrayfire::print_gen("hashdata".to_string(), &hashdata[&0],Some(6));

}
