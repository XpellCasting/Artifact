"""Global class/episode mass and checkpoint selection regression tests."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from app.modulos.clasificador.balancing import EpisodeBalance
from app.modulos.clasificador.evaluation import metrics
from app.modulos.clasificador.training import masked_classification_loss, checkpoint_rank, train_classifier
from app.modulos.clasificador.config import Config
from test_gat_lstm_training import sample


def rows_for_counts(counts):
    rows = []
    for label, count in enumerate(counts):
        for episode in range(count):
            row = sample(label)
            row["metadata"].update(parent_episode_id=f"c{label}-e{episode}",source_id="real")
            # Unequal repetition must not change the total mass of an episode.
            rows.extend(deepcopy(row) for _ in range((episode % 3)+1))
    return rows


class BalancingTests(unittest.TestCase):
    def test_actual_imbalanced_episode_counts_have_equal_total_class_mass(self):
        rows = rows_for_counts([2501,11,38])
        balance = EpisodeBalance(iter(rows))
        self.assertGreater(balance.class_weights[1]/balance.class_weights[0],200)
        contributions = np.zeros(3)
        target_weights = balance.target_weights(rows)
        for index,row in enumerate(rows):
            label = row["y"][0]
            contributions[label] += balance.class_weights[label]*target_weights[index,0]/balance.normalizer
        np.testing.assert_allclose(contributions,len(rows)/3,rtol=1e-6)
        self.assertEqual(balance.episode_support,[2501,11,38])

    def test_single_row_batches_do_not_cancel_rare_episode_weights(self):
        rows = rows_for_counts([20,1,1])
        balance = EpisodeBalance(iter(rows))
        by_class = np.zeros(3)
        for row in rows:
            loss = masked_classification_loss(torch.zeros(1,1,3),torch.tensor([row["y"]]),
                torch.tensor([row["label_mask"]]),torch.tensor(balance.class_weights,dtype=torch.float32),
                torch.tensor(balance.target_weights([row])),normalizer=balance.normalizer)
            by_class[row["y"][0]] += loss.item()/len(rows)
        np.testing.assert_allclose(by_class,np.log(3)/3,rtol=1e-6)

    def test_source_and_class_are_part_of_episode_identity(self):
        rows = [sample(i) for i in range(3)]
        for row in rows:
            row["metadata"]["parent_episode_id"]="same"
        second = deepcopy(rows[1]); second["metadata"]["source_id"]="another-source"
        rows.append(second)
        balance = EpisodeBalance(iter(rows))
        self.assertEqual(balance.episode_support,[1,2,1])
        self.assertTrue((balance.target_weights(rows)==1).all())

    def test_duplicating_windows_does_not_change_dataset_loss_or_gradient(self):
        def evaluate(repeats):
            rows = [sample(0)]*repeats + [sample(1),sample(2)]
            balance = EpisodeBalance(iter(rows))
            parameter = torch.tensor([.2,-.1,.5],requires_grad=True)
            logits = parameter.view(1,1,3).expand(len(rows),1,3)
            loss = masked_classification_loss(logits,torch.tensor([r["y"] for r in rows]),
                torch.tensor([r["label_mask"] for r in rows]),
                torch.tensor(balance.class_weights,dtype=torch.float32),
                torch.tensor(balance.target_weights(rows)),normalizer=balance.normalizer)
            loss.backward()
            return loss,parameter.grad
        a,ga=evaluate(1); b,gb=evaluate(17)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(ga,gb)

    def test_checkpoint_prefers_better_three_class_f1_even_if_loss_is_higher(self):
        normal_only = metrics([dict(true_label=i,scores=[.9,.05,.05]) for i in range(3)],0,0)
        all_classes = metrics([dict(true_label=i,scores=[.8 if k==i else .1 for k in range(3)]) for i in range(3)],0,0)
        self.assertGreater(checkpoint_rank(all_classes,2.0),checkpoint_rank(normal_only,.01))

    def test_supported_fault_types_receive_equal_mass_without_inventing_absent_types(self):
        rows=rows_for_counts([20,2,1])
        balance=EpisodeBalance(iter(rows),labels=("unavailable","bias","spike","drift"),require_all=False)
        weights=balance.target_weights(rows)
        mass=np.zeros(4)
        for index,row in enumerate(rows):
            label=row["y"][0]
            mass[label]+=balance.class_weights[label]*weights[index,0]/balance.normalizer
        np.testing.assert_allclose(mass[:3],len(rows)/3,rtol=1e-6)
        self.assertEqual(mass[3],0)
        self.assertEqual(balance.class_weights[3],0)

    def test_training_publishes_selected_matrix_balance_and_epoch(self):
        rows = rows_for_counts([4,1,1])
        model,scores = train_classifier(rows,rows,Config(max_epochs=2,dropout=0),[3],["continuous"])
        self.assertEqual(scores["balancing"]["version"],"equal_class_episode_v1")
        best = max(scores["history"],key=lambda r:(r["dev_macro_f1"],min(r["dev_recall_by_class"].values()),-r["dev_loss"]))
        self.assertEqual(scores["selected_epoch"],best["epoch"])
        self.assertEqual(scores["dev_macro_f1"],scores["dev_metrics"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
