'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model_output: torch.tensor) -> torch.tensor:
  '''
  Predicts the labels from the output of the model.

  Args:
  -   model_output: the model output [Dim: (N, 15)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''

  predicted_labels = None
  ############################################################################
  # Student code begin
  ############################################################################
  # print(model_output)
  # predicted_labels = torch.ones(model_output.shape[0])
  # print(predicted_labels.shape)
  # print(f'model_output.shape: {model_output.shape}')
  predicted_labels = torch.argmax(model_output, dim=1)
  # print(model_output.type)
  # print(predicted_labels.shape)
  # raise NotImplementedError('predict_labels not implemented')
  #############################################################################
  # Student code end
  ############################################################################

  return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Note: we have initialized the loss_criterion in the model with the sum
  reduction.

  Args:
  -   model: model (which inherits from nn.Module), and contains loss_criterion
  -   model_output: the raw scores output by the net [Dim: (N, 15)]
  -   target_labels: the ground truth class labels [Dim: (N, )]
  -   is_normalize: bool flag indicating that loss should be divided by the
                    batch size
  Returns:
  -   the loss value
  '''
  loss = None

  ############################################################################
  # Student code begin
  ############################################################################
  loss =  model.loss_criterion(model_output, target_labels)
  if is_normalize:
    loss /= model_output.shape[0]
  # print(loss)
  # raise NotImplementedError('compute_loss not implemented')
  ############################################################################
  # Student code end
  ############################################################################

  return loss
