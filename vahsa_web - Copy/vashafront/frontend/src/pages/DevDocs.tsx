import { Link } from "react-router-dom"
import { Code, GitBranch, Key, Zap, ArrowLeft, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function DevDocs() {
  const sections = [
    {
      icon: Key,
      title: "API Authentication",
      description: "Get started with API keys and authentication",
      content: [
        "Sign up for a developer account",
        "Generate your API key from the dashboard",
        "Include the key in your request headers",
        "Rate limits: 1000 requests per hour"
      ]
    },
    {
      icon: Code,
      title: "API Endpoints",
      description: "Available endpoints and their usage",
      content: [
        "POST /api/v1/chat - Send messages to Vasha AI",
        "GET /api/v1/models - List available models",
        "POST /api/v1/completions - Text completions",
        "WebSocket /ws/chat - Real-time chat"
      ]
    },
    {
      icon: GitBranch,
      title: "SDKs & Libraries",
      description: "Official SDKs for popular languages",
      content: [
        "JavaScript/TypeScript SDK",
        "Python SDK",
        "Go SDK",
        "REST API documentation"
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-12">
          <Button variant="outline" asChild className="mb-6">
            <Link to="/" className="flex items-center space-x-2">
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Home</span>
            </Link>
          </Button>
          
          <div className="flex items-center space-x-3 mb-4">
            <div className="h-12 w-12 gradient-primary rounded-lg flex items-center justify-center">
              <Code className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-4xl font-bold">Developer Documentation</h1>
              <p className="text-xl text-muted-foreground mt-2">
                Build with Vasha AI - comprehensive guides and API reference
              </p>
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <Card className="mb-8 shadow-card border-border/40">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5 text-primary" />
              <span>Quick Start</span>
            </CardTitle>
            <CardDescription>
              Get up and running with Vasha AI in minutes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-sm overflow-x-auto">
              <div className="text-primary"># Install the SDK</div>
              <div className="mt-2">npm install vasha-ai-sdk</div>
              <div className="mt-4 text-primary"># Basic usage</div>
              <div className="mt-2">
                <div>import &#123; VashaAI &#125; from 'vasha-ai-sdk'</div>
                <div className="mt-1">const client = new VashaAI(&#123; apiKey: 'your-api-key' &#125;)</div>
                <div className="mt-1">const response = await client.chat('Hello!')</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sections Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {sections.map((section, index) => (
            <Card key={index} className="shadow-card border-border/40 hover:shadow-glow transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <section.icon className="h-5 w-5 text-primary" />
                  <span>{section.title}</span>
                </CardTitle>
                <CardDescription>{section.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {section.content.map((item, itemIndex) => (
                    <li key={itemIndex} className="text-sm text-muted-foreground flex items-center space-x-2">
                      <div className="h-1.5 w-1.5 bg-primary rounded-full" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Resources */}
        <Card className="shadow-card border-border/40">
          <CardHeader>
            <CardTitle>Additional Resources</CardTitle>
            <CardDescription>
              More tools and resources to help you build amazing applications
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid sm:grid-cols-2 gap-4">
              <Button variant="outline" className="justify-between hover:shadow-card transition-shadow duration-300">
                <span>GitHub Repository</span>
                <ExternalLink className="h-4 w-4" />
              </Button>
              <Button variant="outline" className="justify-between hover:shadow-card transition-shadow duration-300">
                <span>Community Discord</span>
                <ExternalLink className="h-4 w-4" />
              </Button>
              <Button variant="outline" className="justify-between hover:shadow-card transition-shadow duration-300">
                <span>Stack Overflow</span>
                <ExternalLink className="h-4 w-4" />
              </Button>
              <Button variant="outline" className="justify-between hover:shadow-card transition-shadow duration-300">
                <span>Status Page</span>
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}