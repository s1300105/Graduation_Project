void CL_AddReliableCommand( const char *cmd ) {
	int		index;

	// if we would be losing an old command that hasn't been acknowledged,
	// we must drop the connection
	if ( clc.reliableSequence - clc.reliableAcknowledge > MAX_RELIABLE_COMMANDS ) {
		Com_Error( ERR_DROP, "Client command overflow" );
	}
	clc.reliableSequence++;
	index = clc.reliableSequence & ( MAX_RELIABLE_COMMANDS - 1 );
	if ( clc.reliableCommands[ index ] ) {
		Z_Free( clc.reliableCommands[ index ] );
	}
	clc.reliableCommands[ index ] = CopyString( cmd );
}