function [d, e, i] = cepsdist_unsync(x, y, fs, param);
%% CEPSDIST_UNSYNC
%% Cepstral distance between two unsynchronized signals 
%%
%% [D, E, I] = CEPSDIST_UNSYNC(X, Y, FS, PARAM) calculates cepstral distance between 
%% two unsynchronized one-dimensional signals specified by X and Y.
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



% Parameter set up
%----------------------------------------------------------------------

timdif = fix(param.timdif * fs);
steps  = [fix([0.01, 0.002] * fs), 1];


% Search with large step
%----------------------------------------------------------------------


offset  = 0;
stp     = steps(1);
num_stp = fix(timdif / stp);

lags = [-num_stp : num_stp] * stp + offset;
lags = lags(find(abs(lags) <= timdif));

[d, e, i] = cepsdist_unsync_step(x, y, fs, param, lags);


% Search with mediam step
%----------------------------------------------------------------------

offset  = lags(i);
stp     = steps(2);
num_stp = fix(steps(1) / stp);

lags = [-num_stp : num_stp] * stp + offset;
lags = lags(find(abs(lags) <= timdif));

[d, e, i] = cepsdist_unsync_step(x, y, fs, param, lags);


% Search with small step
%----------------------------------------------------------------------

offset  = lags(i);
stp     = steps(3);
num_stp = fix(steps(2) / stp);

lags = [-num_stp : num_stp] * stp + offset;
lags = lags(find(abs(lags) <= timdif));

[d, e, i] = cepsdist_unsync_step(x, y, fs, param, lags);

i = lags(i);



%% **********************************************************************
%% [D, E, I] = CEPSDIST_UNSYNC_STEP(X, Y, FS, PARAM, LAGS);
%% **********************************************************************



function [d, e, i] = cepsdist_unsync_step(x, y, fs, param, lags)


% Calculate cepstral distances for every possible time difference.
%----------------------------------------------------------------------

d = zeros(length(lags), 1);
e = zeros(length(lags), 1);

for k = 1 : length(lags)
  if lags(k) < 0
    x2   = x(1 - lags(k) : end);
    y2   = y(1 : end + lags(k));

    [d(k), e(k)] = cepsdist(x2, y2, fs, param);
  else
    x2   = x(1 : end - lags(k));
    y2   = y(1 + lags(k) : end);

    [d(k), e(k)] = cepsdist(x2, y2, fs, param);
  end
end


% Select the minimum value.
%----------------------------------------------------------------------

[d, i] = min(d);
e = e(i);
