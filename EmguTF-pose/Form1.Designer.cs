namespace EmguTF_pose
{
    partial class EmguTF_posewindow
    {
        /// <summary>
        /// Variable nécessaire au concepteur.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Nettoyage des ressources utilisées.
        /// </summary>
        /// <param name="disposing">true si les ressources managées doivent être supprimées ; sinon, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Code généré par le Concepteur Windows Form

        /// <summary>
        /// Méthode requise pour la prise en charge du concepteur - ne modifiez pas
        /// le contenu de cette méthode avec l'éditeur de code.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.button_start = new System.Windows.Forms.Button();
            this.imageBox = new Emgu.CV.UI.ImageBox();
            ((System.ComponentModel.ISupportInitialize)(this.imageBox)).BeginInit();
            this.SuspendLayout();
            // 
            // button_start
            // 
            this.button_start.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.button_start.Location = new System.Drawing.Point(0, 447);
            this.button_start.Name = "button_start";
            this.button_start.Size = new System.Drawing.Size(599, 64);
            this.button_start.TabIndex = 0;
            this.button_start.Text = "Start";
            this.button_start.UseVisualStyleBackColor = true;
            this.button_start.Click += new System.EventHandler(this.button_start_Click);
            // 
            // imageBox
            // 
            this.imageBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.imageBox.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.imageBox.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bicubic;
            this.imageBox.Location = new System.Drawing.Point(0, 0);
            this.imageBox.Name = "imageBox";
            this.imageBox.Size = new System.Drawing.Size(598, 444);
            this.imageBox.TabIndex = 2;
            this.imageBox.TabStop = false;
            // 
            // EmguTF_posewindow
            // 
            this.ClientSize = new System.Drawing.Size(599, 511);
            this.Controls.Add(this.imageBox);
            this.Controls.Add(this.button_start);
            this.Name = "EmguTF_posewindow";
            this.Load += new System.EventHandler(this.EmguTF_posewindow_Load);
            ((System.ComponentModel.ISupportInitialize)(this.imageBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ListBox lb_viewparam;
        private System.Windows.Forms.Button bc_start;
        private System.Windows.Forms.Button button_start;
        private Emgu.CV.UI.ImageBox imageBox;
    }
}

