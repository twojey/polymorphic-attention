"""Famille N — propriétés comparatives Oracle / student.

Pour ces propriétés on requiert deux attentions A_oracle et A_student
(typiquement post-distillation phase 3). Si A_student n'est pas
disponible dans ctx.metadata['student_attn'], on skip cleanly.

Spec : DOC/CATALOGUE §N1-N3.
"""
