function c = realceps(x, flr)
%% REALCEPS
%% Real-valued cepstral coefficients of real-valued sequences
%% 
%% C = REALCEPS(X, FLR) calculates real-valued cepstral coefficients 
%% real-valued sequences specified by X. Each column of Y contains the
%% cepstral coefficients of the corresponding column of X.



% Check input arguments.
%----------------------------------------------------------------------

if nargin < 2
  flr = -100;
end


% Calculate the power spectra of the input frames.
%----------------------------------------------------------------------

pt = 2^nextpow2(size(x, 1));
Px = abs(fft(x, pt));


% Perform flooring.
%----------------------------------------------------------------------

flr = max(Px(:)) * 10^(flr / 20);
Px  = max(Px, flr);


% Calculate the cepstral coefficients.
%----------------------------------------------------------------------

c = real(ifft(log(Px)));
