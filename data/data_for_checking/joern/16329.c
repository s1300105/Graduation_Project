term_report_winsize(term_T *term, int rows, int cols)
{
    if (term->tl_winpty)
	winpty_set_size(term->tl_winpty, cols, rows, NULL);
}