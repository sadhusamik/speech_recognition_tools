function [d, e] = lpcllr(x, y, fs, param);
%% LPCLLR
%% LPC-based log likelihood ratio
%%
%% [D, E] = LPCLLR(X, Y, PARAM) calculates log likelihood ratio of X
%% to Y.
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


% Calculate auto-correlation coefficients and LPC parameters.
%----------------------------------------------------------------------

X = fft(X, 2^nextpow2(2 * frame - 1));
Y = fft(Y, 2^nextpow2(2 * frame - 1));

Rx = ifft(abs(X).^2);
Rx = Rx ./ frame; 
Rx = real(Rx);
Ry = ifft(abs(Y).^2);
Ry = Ry ./ frame; 
Ry = real(Ry);

[Ax, Ex] = levinson(Rx, param.lpcorder);
Ax       = real(Ax');
[Ay, Ey] = levinson(Ry, param.lpcorder);
Ay       = real(Ay');


% Calculate LLR. 
%----------------------------------------------------------------------

ds = zeros(num_frame, 1);

for n = 1 : num_frame
  R = toeplitz(Ry(1 : param.lpcorder + 1, n));

  num = Ax(:, n)' * R * Ax(:, n);
  den = Ay(:, n)' * R * Ay(:, n);
  
  ds(n) = log(num / den);
end

%% outlier removal
ds = sort(ds);
ds = ds(1 : ceil(num_frame * 0.95));
ds = min(ds, 2);
ds = max(ds, 0);

d = mean(ds);
e = median(ds);
