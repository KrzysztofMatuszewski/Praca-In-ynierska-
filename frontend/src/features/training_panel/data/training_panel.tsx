import { useState } from 'react'
import { IconBrain, IconColumns, IconCalendar, IconListNumbers, IconMaximize } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Toaster } from '@/components/ui/toaster'

export default function ModelTraining() {
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState({
    source: 'hids',
    batch_size: 50,
    columns_to_use: ['agent.id', 'rule.description'],
    relative_from: '',
    relative_to: '',
    epochs: 50,
    max_size: 10000
  })

  // Set default dates (from: yesterday, to: today)
  useState(() => {
    const today = new Date()
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)
    
    setFormData(prev => ({
      ...prev,
      relative_from: formatDateForInput(yesterday),
      relative_to: formatDateForInput(today)
    }))
  })

  function formatDateForInput(date: Date) {
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}T${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
  }

  const handleInputChange = (e: { target: { name: any; value: any } }) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleColumnsChange = (e: { target: { value: string } }) => {
    const columns = e.target.value.split(',')
      .map(col => col.trim())
      .filter(col => col !== '')
    
    setFormData(prev => ({
      ...prev,
      columns_to_use: columns
    }))
  }

  const handleRadioChange = (value: any) => {
    setFormData(prev => ({
      ...prev,
      source: value
    }))
  }

  const handleNumberChange = (e: { target: { name: any; value: any } }) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: parseInt(value, 10)
    }))
  }

  const handleSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Here you would normally send the data to your API
      // For demo purposes, we'll just simulate a successful API call
      console.log('Submitting form data:', formData)
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Show success message
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Header>
        <Search />
        <div className='ml-auto flex items-center gap-4'>
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        <div className="flex flex-col space-y-6 max-w-3xl mx-auto py-6">
          <div>
            <h1 className='text-2xl font-bold tracking-tight'>
              Model Training Configuration
            </h1>
            <p className="text-muted-foreground mt-2">
              Configure and start the training process for anomaly detection models.
            </p>
          </div>
          <Separator className='shadow' />
          
          <form onSubmit={handleSubmit} className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle>Data Source Selection</CardTitle>
                <CardDescription>
                  Choose which type of data to use for training the model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <RadioGroup 
                  defaultValue={formData.source} 
                  className="flex space-x-4"
                  onValueChange={handleRadioChange}
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="hids" id="hids" />
                    <Label htmlFor="hids">HIDS (Host Intrusion Detection)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="nids" id="nids" />
                    <Label htmlFor="nids">NIDS (Network Intrusion Detection)</Label>
                  </div>
                </RadioGroup>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Data Selection Parameters</CardTitle>
                <CardDescription>
                  Define the time range and maximum number of records to use for training
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="relative_from" className="flex items-center">
                      <IconCalendar className="mr-2 h-4 w-4" />
                      Start Date and Time
                    </Label>
                    <Input 
                      id="relative_from"
                      name="relative_from"
                      type="datetime-local"
                      value={formData.relative_from}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="relative_to" className="flex items-center">
                      <IconCalendar className="mr-2 h-4 w-4" />
                      End Date and Time
                    </Label>
                    <Input 
                      id="relative_to"
                      name="relative_to"
                      type="datetime-local"
                      value={formData.relative_to}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="max_size" className="flex items-center">
                    <IconMaximize className="mr-2 h-4 w-4" />
                    Maximum Records to Process
                  </Label>
                  <Input 
                    id="max_size"
                    name="max_size"
                    type="number"
                    min="100"
                    value={formData.max_size}
                    onChange={handleNumberChange}
                    required
                  />
                  <p className="text-sm text-muted-foreground">
                    The maximum number of records to retrieve from the data source.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Configuration</CardTitle>
                <CardDescription>
                  Configure training parameters for the autoencoder model
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="epochs" className="flex items-center">
                      <IconListNumbers className="mr-2 h-4 w-4" />
                      Number of Epochs
                    </Label>
                    <Input 
                      id="epochs"
                      name="epochs"
                      type="number"
                      min="1"
                      max="1000"
                      value={formData.epochs}
                      onChange={handleNumberChange}
                      required
                    />
                    <p className="text-sm text-muted-foreground">
                      How many training cycles to run.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="batch_size" className="flex items-center">
                      <IconBrain className="mr-2 h-4 w-4" />
                      Batch Size
                    </Label>
                    <Input 
                      id="batch_size"
                      name="batch_size"
                      type="number"
                      min="1"
                      max="1000"
                      value={formData.batch_size}
                      onChange={handleNumberChange}
                      required
                    />
                    <p className="text-sm text-muted-foreground">
                      Number of samples processed before model update.
                    </p>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="columns_to_use" className="flex items-center">
                    <IconColumns className="mr-2 h-4 w-4" />
                    Columns to Use
                  </Label>
                  <Textarea 
                    id="columns_to_use"
                    value={formData.columns_to_use.join(', ')}
                    onChange={handleColumnsChange}
                    placeholder="Enter column names separated by commas"
                    className="min-h-20"
                    required
                  />
                  <p className="text-sm text-muted-foreground">
                    Specify which data columns to include in the training process (comma-separated).
                  </p>
                </div>
              </CardContent>
              <CardFooter className="flex justify-end">
                <Button type="submit" disabled={loading} className="w-full md:w-auto">
                  {loading ? "Starting Training..." : "Start Model Training"}
                </Button>
              </CardFooter>
            </Card>
          </form>
        </div>
        <Toaster />
      </Main>
    </>
  )
}