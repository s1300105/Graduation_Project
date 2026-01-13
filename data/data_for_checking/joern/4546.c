    void operator = (const IniSection &s)
	{
	    if (&s == this)
	    {
		return;
	    } 
	    IniBase::operator = (s);
	    ip = s.ip;
	    end_comment = s.end_comment;
            is_private = s.is_private;
            rewrite_by = s.rewrite_by;
	    container = s.container;

	    reindex ();
	}