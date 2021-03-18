%% process_RealData.m
%%
%% Simply copy each speech signal from RealData to an output directory.
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
rooms   = {'room1'};
taskdir = '../taskfiles/1ch';

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist = dists{i1};
    room = rooms{i2};
    
    listname = fullfile(taskdir, ['RealData_dt_for_1ch_', dist, '_', room, '_A']);

    iroot = fullfile(download_from_ldc, 'MC_WSJ_AV_Dev');
    oroot = '../output/RealData';
    
    copyfiles(listname, iroot, oroot);
  end
end
