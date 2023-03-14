function J = numerical_jacobian(x, f, vargs, delta)
% numerical_jacobian -- numerically estimate the Jacobian matrix of a function
%
% J = numerical_jacobian(x, f, vargs, delta)
% Numerically estimates the Jacobian of a function f about vectors in x by
% calling f(x, vargs{:}) with delta pertubations of x
%
% Arguments
%
% x: a MxK matrix containing K linearization points
% f: function handle to the function to be called, which must accept Mx1 vectors
%    and return Nx1 vectors
% vargs: a cell array of additional arguments to f (optional)
% delta: a scalar of Mx1 vector containing the perturbation distance for x 
%        (defaults to 1e-8)
%
% Return Values
%
% J: an NxMxK matrix containing the partial derivatives of the input and out
%    elements of f
%
% Author: thedon@google.com (L. Donnie Smith)

if nargin < 4
  delta = 1e-8;
  if nargin < 3
    vargs = {};
  end
end

y = f(x, vargs{:});
N = size(y, 1);

K = size(x);
M = K(1);
K(1) = [];

if isscalar(delta)
  delta = repmat(delta, [M 1]);
end

% Compute the Jacobian by independently perturbing each dimension of x,
% subtracting y, and dividing by delta
y_delta = zeros([N M K]);
for idx = 1:M
  x_delta = x;
  x_delta(idx,:) = x_delta(idx,:) + delta(idx);
  y_delta(:,idx,:) = reshape(f(x_delta, vargs{:}), N, 1, []);
end

J = (y_delta-repmat(reshape(y, [N 1 K]), [1 M 1]))./repmat(delta', [N 1 K]);
