function score_mat = fastXML_predict( tst_X_Xf, model_dir, param )

    file_tst_X_Xf = tempname;
    write_text_mat(tst_X_Xf,file_tst_X_Xf);
    
    file_score_mat = tempname;
   
	clear tst_X_Xf;
 
    cmd = sprintf('fastXML_predict %s %s %s %s', file_tst_X_Xf, file_score_mat, model_dir, get_arguments(param));
    if isunix
        cmd = ['./' cmd];
    end
    
	system(cmd);

    score_mat = read_text_mat(file_score_mat);
end

function args = get_arguments(param)
	args = ' ';
	
	if isfield(param,'num_thread')
		args = sprintf(' %s -T %d',args,param.num_thread);
	end

	if isfield(param,'start_tree')
		args = sprintf(' %s -s %d',args,param.start_tree);
	end

	if isfield(param,'num_tree')
		args = sprintf(' %s -t %d',args,param.num_tree);
	end
end

