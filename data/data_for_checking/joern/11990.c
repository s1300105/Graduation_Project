pci_get_cfgdata16(struct pci_vdev *dev, int offset)
{
	if ((offset > PCI_REGMAX - 1) || (offset & 1) != 0) {
		pr_err("%s: out of range of PCI config space!\n", __func__);
		return 0xffff;
	}
	return (*(uint16_t *)(dev->cfgdata + offset));
}