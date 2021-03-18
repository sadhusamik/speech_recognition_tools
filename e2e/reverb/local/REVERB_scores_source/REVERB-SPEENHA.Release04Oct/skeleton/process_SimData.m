%% process_SimData.m
%%
%% Simply copy each speech signal from SimData to an output directory.
%% This may be used as a skeleton for your own algorithm. Single 
%% microphone is assumed. 
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



clear all;



% download_from_ldc = '/directory/where/you/have/stored/data/obtained/from/LDC';

if ~exist('download_from_ldc', 'var')
  fprintf('Uncomment download_from_ldc!\n');
  return;
end


% File names
%----------------------------------------------------------------------

addpath ./prog;

dists  = {'far', 'near'};
rooms   = {'room1', 'room2', 'room3'};
taskdir = '../taskfiles/1ch';

iroot = fullfile(download_from_ldc, 'REVERB_WSJCAM0_dt/data');
oroot = '../output/SimData';

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist = dists{i1};
    room = rooms{i2};
    
    listname = fullfile(taskdir, ['SimData_dt_for_1ch_', dist, '_', room, '_A']);
    
    copyfiles(listname, iroot, oroot);
  end
end
