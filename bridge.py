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

    def is_linked(self):
        """
        Returns whether this environment is linked to both the main and adversarial agent
        """
        return self.adv_agent is not None and self.prot_agent is not None
