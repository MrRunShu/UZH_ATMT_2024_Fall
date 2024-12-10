import matplotlib.pyplot as plt

# data
beam_sizes = [1, 5, 10, 15, 20, 25]
bleu_scores = [17.1, 20.0, 22.2, 22.1, 21.8, 21.4]
brevity_penalties = [1.000, 1.000, 1.000, 0.949, 0.864, 0.790]

fig, ax1 = plt.subplots(figsize=(8, 6)) 

# BLEU 
ax1.set_xlabel('Beam Size (k)', fontsize=12)
ax1.set_ylabel('BLEU Score', color='tab:blue', fontsize=12)
ax1.plot(beam_sizes, bleu_scores, marker='o', color='tab:blue', label='BLEU Score')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Brevity Penalty (BP)
ax2 = ax1.twinx()
ax2.set_ylabel('Brevity Penalty (BP)', color='tab:orange', fontsize=12)
ax2.plot(beam_sizes, brevity_penalties, marker='x', color='tab:orange', label='Brevity Penalty')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout(rect=[0, 0, 1, 0.95])  
plt.title('BLEU Score and Brevity Penalty vs Beam Size', fontsize=14, pad=20)  # 增加标题间距

# Save and show plot
plt.savefig('visualize_BLEU.png', dpi=300, bbox_inches='tight')
plt.show()
