import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model


class SSRotate3D(Model):

	def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, tolerance=0.1):
		super(SSRotate3D, self).__init__(ent_tot, rel_tot)
		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = dim * 3
		self.dim_r = dim * 4

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
		self.ent_radius = nn.Embedding(self.ent_tot, 1)  # 1 for the sphere radius

		# radius
		self.ent_radius_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
			requires_grad=False
		)

		self.tolerance = tolerance

		nn.init.uniform_(
			tensor=self.ent_radius.weight.data,
			a=-self.ent_radius_range.item(),
			b=self.ent_radius_range.item()
		)

		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
			requires_grad=False
		)

		nn.init.uniform_(
			tensor=self.ent_embeddings.weight.data,
			a=-self.ent_embedding_range.item(),
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
			requires_grad=False
		)

		nn.init.uniform_(
			tensor=self.rel_embeddings.weight.data,
			a=-self.rel_embedding_range.item(),
			b=self.rel_embedding_range.item()
		)

		nn.init.ones_(
			tensor=self.rel_embeddings.weight.data[:, self.dim_e:self.dim_r]
		)

		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False


	def _calc(self, head, tail, rel, head_radius, tail_radius, mode):
		pi = self.pi_const
		# print("head",head.shape)  # torch.Size([70746, 600])
		head_i, head_j, head_k = torch.chunk(head, 3, dim=1)
		beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=1)
		tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=1)

		bias = torch.abs(bias)

		# Make phases of relations uniformly distributed in [-pi, pi]
		beta_1 = beta_1 / (self.ent_embedding_range.item() / pi)
		beta_2 = beta_2 / (self.ent_embedding_range.item() / pi)
		theta = theta / (self.ent_embedding_range.item() / pi)
		cos_theta = torch.cos(theta)
		sin_theta = torch.sin(theta)

		# Obtain representation of the rotation axis
		rel_i = torch.cos(beta_1)
		rel_j = torch.sin(beta_1) * torch.cos(beta_2)
		rel_k = torch.sin(beta_1) * torch.sin(beta_2)

		C = rel_i * head_i + rel_j * head_j + rel_k * head_k
		C = C * (1 - cos_theta)

		# Rotate the head entity
		new_head_i = head_i * cos_theta + C * rel_i + sin_theta * (rel_j * head_k - head_j * rel_k)
		new_head_j = head_j * cos_theta + C * rel_j - sin_theta * (rel_i * head_k - head_i * rel_k)
		new_head_k = head_k * cos_theta + C * rel_k + sin_theta * (rel_i * head_j - head_i * rel_j)

		score_i = new_head_i * bias - tail_i
		score_j = new_head_j * bias - tail_j
		score_k = new_head_k * bias - tail_k

		score = torch.stack([score_i, score_j, score_k], dim=0)
		head_radius = torch.nn.functional.relu(torch.norm(head_radius, 1, -1))	# torch.Size([70746])
		tail_radius = torch.nn.functional.relu(torch.norm(tail_radius, 1, -1))
		score = score.norm(dim=0).sum(dim=-1)
		if mode == 'head_batch':
			score, max_idxs = torch.max(torch.stack((score-head_radius-tail_radius, -self.tolerance * head_radius)), 0)
			score = score + self.tolerance * head_radius
		else:
			score, max_idxs = torch.max(torch.stack((score-head_radius-tail_radius, -self.tolerance * tail_radius)), 0)
			score = score + self.tolerance * tail_radius	
		return score


	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		# add head_radius tail_radius
		head_radius = self.ent_radius(batch_h)
		tail_radius = self.ent_radius(batch_t)
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self.margin - self._calc(h, t, r, head_radius, tail_radius, mode)
		return score


	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()


	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		radius = (self.ent_radius(batch_h) + self.ent_radius(batch_t))/2	# add radius
		regul = (torch.mean(h ** 2) +
				torch.mean(t ** 2) +
				torch.mean(r ** 2)) / 3
		return regul

	def helper_dist(self, head, tail, rel, head_radius, tail_radius, mode):
		pi = self.pi_const
		# print("head",head.shape)  # torch.Size([70746, 600])
		head_i, head_j, head_k = torch.chunk(head, 3, dim=1)
		beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=1)
		tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=1)

		bias = torch.abs(bias)

		# Make phases of relations uniformly distributed in [-pi, pi]
		beta_1 = beta_1 / (self.ent_embedding_range.item() / pi)
		beta_2 = beta_2 / (self.ent_embedding_range.item() / pi)
		theta = theta / (self.ent_embedding_range.item() / pi)
		cos_theta = torch.cos(theta)
		sin_theta = torch.sin(theta)

		# Obtain representation of the rotation axis
		rel_i = torch.cos(beta_1)
		rel_j = torch.sin(beta_1) * torch.cos(beta_2)
		rel_k = torch.sin(beta_1) * torch.sin(beta_2)

		C = rel_i * head_i + rel_j * head_j + rel_k * head_k
		C = C * (1 - cos_theta)

		# Rotate the head entity
		new_head_i = head_i * cos_theta + C * rel_i + sin_theta * (rel_j * head_k - head_j * rel_k)
		new_head_j = head_j * cos_theta + C * rel_j - sin_theta * (rel_i * head_k - head_i * rel_k)
		new_head_k = head_k * cos_theta + C * rel_k + sin_theta * (rel_i * head_j - head_i * rel_j)

		score_i = new_head_i * bias - tail_i
		score_j = new_head_j * bias - tail_j
		score_k = new_head_k * bias - tail_k

		score = torch.stack([score_i, score_j, score_k], dim=0)
		head_radius = torch.nn.functional.relu(torch.norm(head_radius, 1, -1))	# torch.Size([70746])
		tail_radius = torch.nn.functional.relu(torch.norm(tail_radius, 1, -1))
		score = score.norm(dim=0).sum(dim=-1)
		score, max_idxs = torch.max(torch.stack((score-head_radius-tail_radius, torch.zeros(score.shape[0]).cuda())), 0)
		return score


	def dist(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		# add head_radius tail_radius
		head_radius = self.ent_radius(batch_h)
		tail_radius = self.ent_radius(batch_t)
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self.margin - self._calc(h, t, r, head_radius, tail_radius, mode)
		return score


	def helper_for_test_score(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()