function [d, e] = cepsdist(x, y, fs, param);
%% CEPSDIST
%% Cepstral distance between two signals
%%
%% [D, E] = CEPSDIST(X, Y, FS, PARAM) calculates cepstral distance between 
%% two one-dimensional signals specified by X and Y.
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



% Calculate the number of frames.
%----------------------------------------------------------------------

if length(x) > length(y)
  x = x(1 : length(y));
else
  y = y(1 : length(x));
end

%% Normalization
if ~strcmp(param.cmn, 'y')
  x = x / sqrt(sum(x.^2));
  y = y / sqrt(sum(y.^2));
end

frame = fix(param.frame * fs);
shift = fix(param.shift * fs);

num_sample = length(x);
num_frame  = fix((num_sample - frame + shift) / shift);


% Break up the signals into frames.
%----------------------------------------------------------------------

win = window(param.window, frame);

idx = repmat((1 : frame)', 1, num_frame) + ...
      repmat((0 : num_frame - 1) * shift, frame, 1);

X = bsxfun(@times, x(idx), win);
Y = bsxfun(@times, y(idx), win);


% Apply the cepstrum analysis.
%----------------------------------------------------------------------

ceps_x = realceps(X);
ceps_y = realceps(Y);

ceps_x = ceps_x(1 : param.order + 1, :);
ceps_y = ceps_y(1 : param.order + 1, :);


% Perform cepstral mean normalization.
%----------------------------------------------------------------------

if strcmp(param.cmn, 'y')
  ceps_x = bsxfun(@minus, ceps_x, mean(ceps_x, 2));
  ceps_y = bsxfun(@minus, ceps_y, mean(ceps_y, 2));
end


% Calculate the cepstral distances
%----------------------------------------------------------------------

err = (ceps_x - ceps_y) .^2;
ds  = 10 / log(10) * sqrt(2 * sum(err(2 : end, :), 1) + err(1, :));
ds  = min(ds, 10);
ds  = max(ds, 0);

d = mean(ds);
e = median(ds);

