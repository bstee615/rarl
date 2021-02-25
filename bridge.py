class Bridge:
    """
    Bridge between main agent and adversarial agent.
    """

    def __init__(self):
        super().__init__()
        self.adv_agent = None
        self.prot_agent = None

    def link_agents(self, prot_agent, adv_agent):
        """
        Link main_agent and adv_agent. These are the two agents which will be taking actions.
        """
        self.prot_agent = prot_agent
        self.adv_agent = adv_agent

    def other(self, im_protagonist):
        return self.adv_agent if im_protagonist else self.prot_agent
