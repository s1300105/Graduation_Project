    IniSection (const IniParser *p)
	: IniBase (-1),
	  ip (p),
	  end_comment (), is_private(false), rewrite_by(-1),
	  container (), ivalues (), isections ()
	    {}