class Bridge:
    """
    Bridge between main agent and adversarial agent.
    """

    def __init__(self):
        super().__init__()
        self.adv_agent = None
        self.main_agent = None

    def link_agents(self, main_agent, adv_agent):
        """
        Link main_agent and adv_agent. These are the two agents which will be taking actions.
        """
        self.main_agent = main_agent
        self.adv_agent = adv_agent

    def is_linked(self):
        """
        Returns whether this environment is linked to both the main and adversarial agent
        """
        return self.adv_agent is not None and self.main_agent is not None
