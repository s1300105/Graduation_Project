    IniSection (const IniParser *p)
	: IniBase (-1),
	  ip (p),
	  end_comment (), rewrite_by(-1),
	  container (), ivalues (), isections ()
	    {}