function copyfiles(listname, iroot, oroot);
%% COPYFILES
%% Simply copy each speech signal from SimData to an output directory.
%% This may be used as a skeleton for your own algorithm. Single 
%% microphone is assumed. 
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



% Create a list of file names.
%----------------------------------------------------------------------

fid = fopen(listname, 'r');

num_file = 0;
namelist = cell(10000, 1);

while ~feof(fid)
  num_file           = num_file + 1;
  namelist{num_file} = fgetl(fid);  
end

fclose(fid);

namelist = namelist(1 : num_file);


% Process each file.
%----------------------------------------------------------------------

for fidx = 1 : num_file  
  ifname = fullfile(iroot, namelist{fidx});
  ofname = fullfile(oroot, namelist{fidx});
  
  odir = fileparts(ofname);
  cmd = ['mkdir -p -v ', odir];
  system(cmd);
  
  %% Show progress.
  fprintf('[%04d/%04d]] %s\n', fidx, num_file, ifname);
  fprintf('    -> %s\n', ofname);

  %% Load a single-channel corrupted speech signal in X.
  [x, fs, qbit] = wavread(ifname);
  
  %% Enhance X and store the result in Y.
  %% Use your own version of SPENH. 
  y = spenh(x);
  
  %% Then, store the content of Y in a file specified by OFNAME.
  wavwrite(y, fs, qbit, ofname);
  
end

fprintf('\n');





function y = spenh(x);
%% SPENH
%% This simply reads the input and copies the content to return variable Y. 
%% Replace this function with your speecn enhancement algorithm. 

y = x;
