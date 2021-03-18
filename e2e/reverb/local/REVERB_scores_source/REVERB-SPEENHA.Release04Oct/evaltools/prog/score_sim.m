function score_sim(name, resdir, tgtlist, reflist, tgtroot, refroot, srmrdir, pesqexe);
%% SCORE_SIM
%% Evaluate speech enhancement results for SimData. This is
%% designed for use in the REVERB challenge. 
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



% Set up
%----------------------------------------------------------------------

addpath(srmrdir);
warning('off', 'MATLAB:dispatcher:pathWarning');

resdir = fullfile(resdir, 'work');

cmd = ['mkdir -p -v ', resdir];
system(cmd);

%% Cepstral distance
param_cd = struct('frame' , 0.025   , ...
		  'shift' , 0.01    , ...
		  'window', @hanning, ...
		  'order' , 24      , ...
		  'timdif', 0.0     , ...
		  'cmn'   , 'y');

%% Log likelihood ratio
param_llr = struct('frame' , 0.025, ...
		   'shift' , 0.01, ...
		   'window', @hanning, ...
		   'lpcorder', 12);

%% Frequency-weighted segmental SNR
param_fwsegsnr = struct('frame'  , 0.025, ...
			'shift'  , 0.01, ...
			'window' , @hanning, ...
			'numband', 23);


% List files to be evaluated.
%----------------------------------------------------------------------

num_file = 0;

tgt = cell(10000, 1);
ref = cell(10000, 1);

tgtfid = fopen(tgtlist);
reffid = fopen(reflist);

while ~feof(tgtfid)
  num_file = num_file + 1;  

  tgt{num_file} = strtrim(fgetl(tgtfid));
  ref{num_file} = strtrim(fgetl(reffid));
end

fclose(tgtfid);
fclose(reffid);

tgt = tgt(1 : num_file);
ref = ref(1 : num_file);


% Create a result file.
%----------------------------------------------------------------------

fid  = fopen(fullfile(resdir, name), 'w');
fids = [1, fid];


% Evaluate each file.
%----------------------------------------------------------------------

for m = 1 : 2
  fprintf(fids(m), '%s\n', datestr(now, 'mmmm dd, yyyy  HH:MM:SS AM'));
  fprintf(fids(m), '%s\n\n', fullfile(pwd, mfilename));
  
  fprintf(fids(m), 'TARGET LIST    : %s\n'  , tgtlist);
  fprintf(fids(m), 'TARGET ROOT    : %s\n'  , tgtroot);
  fprintf(fids(m), 'REFEFENCE LIST : %s\n'  , reflist);
  fprintf(fids(m), 'REFERENCE ROOT : %s\n\n', refroot);
  
  fprintf(fids(m), 'Cepstrum parameters:\n');
  fprintf(fids(m), 'FRAME SIZE     = %d ms\n', param_cd.frame * 1e3);
  fprintf(fids(m), 'SHIFT SIZE     = %d ms\n', param_cd.shift * 1e3);
  fprintf(fids(m), 'WINDOW         = %s\n'   , func2str(param_cd.window));
  fprintf(fids(m), 'CEPS ORDER     = %d\n'   , param_cd.order);
  fprintf(fids(m), 'MAX TIME DIFF  = %d ms\n', param_cd.timdif * 1e3);
  fprintf(fids(m), 'MEAN NORMALIZE = %s\n\n' , param_cd.cmn);
  
  fprintf(fids(m), 'Log likelihood ratio parameters:\n');
  fprintf(fids(m), 'FRAME SIZE     = %d ms\n', param_llr.frame * 1e3);
  fprintf(fids(m), 'SHIFT SIZE     = %d ms\n', param_llr.shift * 1e3);
  fprintf(fids(m), 'WINDOW         = %s\n'   , func2str(param_llr.window));
  fprintf(fids(m), 'LPC ORDER     = %d\n'    , param_llr.lpcorder);

  fprintf(fids(m), 'Frequency-weighted segmental SNR parameters:\n');
  fprintf(fids(m), 'FRAME SIZE     = %d ms\n', param_fwsegsnr.frame * 1e3);
  fprintf(fids(m), 'SHIFT SIZE     = %d ms\n', param_fwsegsnr.shift * 1e3);
  fprintf(fids(m), 'WINDOW         = %s\n'   , func2str(param_fwsegsnr.window));
  fprintf(fids(m), 'MEL BANDS      = %s\n'   , num2str(param_fwsegsnr.numband));

  fprintf(fids(m), 'SRMR directory:\n');
  fprintf(fids(m), '%s\n', srmrdir);
  
  if ~isempty(pesqexe)
    fprintf(fids(m), 'PESQ executable:\n');
    fprintf(fids(m), '%s\n', pesqexe);
  end
  fprintf(fids(m), '\n');
  
  fprintf(fids(m), '----------------------------------------------------------------------\n');
  fprintf(fids(m), 'Individual results\n\n');
end

cd_mean   = zeros(num_file, 1);
cd_med    = zeros(num_file, 1);
srmr_mean = zeros(num_file, 1);
llr_mean  = zeros(num_file, 1);
llr_med   = zeros(num_file, 1);
snr_mean  = zeros(num_file, 1);
snr_med   = zeros(num_file, 1);

if ~isempty(pesqexe)
  pesq_mean = zeros(num_file, 1);
end

for k = 1 : num_file
  tgtname = fullfile(tgtroot, tgt{k});
  refname = fullfile(refroot, ref{k});
  
  for m = 1 : 2
    fprintf(fids(m), '[%04d of %04d]\n', k, num_file);
    fprintf(fids(m), 'TARGET   : %s\n' , tgtname);
    fprintf(fids(m), 'REFERENCE: %s\n' , refname);
  end

  %% Load signals.
  [y, fs] = audioread(tgtname);
  [x, fs] = audioread(refname);
  
  if length(y) > length(x)
    y = y(1 : length(x));
  elseif length(y) < length(x)
    x = x(1 : length(y));
  else
    ;
  end
  

  %%%%%%%%%%%%%%%%%%%%%%%
  %% Cepstral distance %%
  %%%%%%%%%%%%%%%%%%%%%%%
  
  [cd_mean(k), cd_med(k), timdif] = cepsdist_unsync(x, y, fs, param_cd);

  for m = 1 : 2
    fprintf(fids(m), '\tTIMEDIFF       : %6d samples (%.3f s)\n', timdif, timdif / fs);
    fprintf(fids(m), '\tCEPSDIST (MEAN): %6.2f dB\n', cd_mean(k));
    fprintf(fids(m), '\tCEPSDIST (MED) : %6.2f dB\n', cd_med(k));
  end
  
  if timdif < 0
    x = x(1 - timdif : end);
    y = y(1 : end + timdif);
  else
    x = x(1 : end - timdif);
    y = y(1 + timdif : end);
  end
  
  
  %%%%%%%%%%
  %% SRMR %%
  %%%%%%%%%%

  srmr_mean(k) = SRMR(tgtname);
  
  for m = 1 : 2
    fprintf(fids(m), '\tSRMR           : %6.2f\n', srmr_mean(k));  
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Log likelihood ratio %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  
  [llr_mean(k), llr_med(k)] = lpcllr(y, x, fs, param_llr);

  for m = 1 : 2
    fprintf(fids(m), '\tLLR      (MEAN): %6.2f\n', llr_mean(k));  
    fprintf(fids(m), '\tLLR      (MED) : %6.2f\n', llr_med(k));  
  end

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Frequency-weighted segmental SNR %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  [snr_mean(k), snr_med(k)] = fwsegsnr(y, x, fs, param_fwsegsnr);
  
  for m = 1 : 2
    fprintf(fids(m), '\tFWSEGSNR (MEAN): %6.2f dB\n', snr_mean(k));  
    fprintf(fids(m), '\tFWSEGSNR (MED) : %6.2f dB\n', snr_med(k));  
  end
  
  
  %%%%%%%%%%
  %% PESQ %%
  %%%%%%%%%%

  if ~isempty(pesqexe)
    pesq_mean(k) = calcpesq(tgtname, refname, pesqexe);
    
    for m = 1 : 2
      fprintf(fids(m), '\tPESQ           : %6.2f\n', pesq_mean(k));  
    end
  end
  
  for m = 1 : 2
    fprintf(fids(m), '\n');
  end
end


% Print a summary.
%----------------------------------------------------------------------

avg_cd_mean   = mean(cd_mean);
avg_cd_med    = mean(cd_med);
avg_srmr_mean = mean(srmr_mean);
avg_llr_mean  = mean(llr_mean);
avg_llr_med   = mean(llr_med);
avg_snr_mean  = mean(snr_mean);
avg_snr_med   = mean(snr_med);

if ~isempty(pesqexe)
  avg_pesq_mean = mean(pesq_mean);
end

for m = 1 : 2
  fprintf(fids(m), '----------------------------------------------------------------------\n');
  fprintf(fids(m), 'Summary\n\n');
  fprintf(fids(m), 'AVG CEPSDIST (MEAN): %6.2f dB\n', avg_cd_mean);
  fprintf(fids(m), 'AVG CEPSDIST (MED) : %6.2f dB\n', avg_cd_med);
  fprintf(fids(m), 'AVG SRMR           : %6.2f\n'   , avg_srmr_mean);
  fprintf(fids(m), 'AVG LLR      (MEAN): %6.2f\n'   , avg_llr_mean);
  fprintf(fids(m), 'AVG LLR      (MED) : %6.2f\n'   , avg_llr_med);
  fprintf(fids(m), 'AVG FWSEGSNR (MEAN): %6.2f dB\n', avg_snr_mean);
  fprintf(fids(m), 'AVG FWSEGSNR (MED) : %6.2f dB\n', avg_snr_med);
  
  if ~isempty(pesqexe)
    fprintf(fids(m), 'AVG PESQ           : %6.2f\n'   , avg_pesq_mean);
  end
end

fclose(fid);
