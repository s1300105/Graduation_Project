MemoryManager::MemoryManager(u32 physical_address_for_kernel_page_tables)
{
    CPUID id(0x80000001);
    m_has_nx_support = (id.edx() & (1 << 20)) != 0;

    m_kernel_page_directory = PageDirectory::create_at_fixed_address(PhysicalAddress(physical_address_for_kernel_page_tables));
    for (size_t i = 0; i < 4; ++i) {
        m_low_page_tables[i] = (PageTableEntry*)(physical_address_for_kernel_page_tables + PAGE_SIZE * (5 + i));
        memset(m_low_page_tables[i], 0, PAGE_SIZE);
    }

    initialize_paging();

    kprintf("MM initialized.\n");
}